from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from skimage import morphology

from .crossing import regularize_crossings
from .image_ops import load_mask
from .skeleton import junction_pixels, prune_spurs, trace_skeleton
from .stitch import stitch_segments
from .utils import (
    resample_by_arclength,
    rdp,
    sanitize_segment,
    smooth_moving_average,
    split_large_jumps,
)


def process_one(
    path: Path,
    *,
    max_dim: int = 1600,
    spur: int = 15,
    eps: float = 1.0,
    resample_ppx: float = 0.30,
    resample_min: int = 50,
    resample_max: int = 900,
    smooth_k: int = 3,
    alpha_thresh: int = 8,
    max_jump_mult: float = 1.75,
    max_join_mult: float = 0.45,
    angle_tol: float = 10.0,
):
    # Work small for geometry, then upscale once to match original pixel grid.
    mask_small, (img_w, img_h), (scaled_w, _scaled_h) = load_mask(
        path, max_dim, alpha_thresh
    )

    # Closing fixes tiny antialias gaps; remove specks to avoid spurious branches.
    mask_small = morphology.binary_closing(mask_small, morphology.square(3))
    mask_small = morphology.remove_small_objects(mask_small, 64)

    skeleton, distance = morphology.medial_axis(mask_small, return_distance=True)
    skeleton = prune_spurs(skeleton, max_len=spur)

    segments = trace_skeleton(skeleton)
    if not segments:
        return img_w, img_h, [], 0.0

    radii = distance[skeleton]
    typical_w_small = 2.0 * float(np.median(radii)) if radii.size else 6.0
    scale = img_w / float(scaled_w)
    width_px = max(1.0, typical_w_small * scale)

    processed_small: List[List[Tuple[float, float]]] = []
    for segment in segments:
        seg2 = rdp(segment, eps) if eps else list(segment)
        length_small = sum(
            math.hypot(
                seg2[i + 1][0] - seg2[i][0],
                seg2[i + 1][1] - seg2[i][1],
            )
            for i in range(len(seg2) - 1)
        )
        n_samples = (
            max(
                resample_min,
                min(resample_max, int((length_small * scale) * resample_ppx)),
            )
            if length_small > 0
            else resample_min
        )
        seg2 = resample_by_arclength(seg2, n_samples)
        if smooth_k:
            seg2 = smooth_moving_average(seg2, smooth_k)
        processed_small.append(seg2)

    junctions = junction_pixels(skeleton, min_deg=4)
    processed_small = regularize_crossings(processed_small, junctions, steps=6)

    runs: List[List[Tuple[float, float]]] = []
    max_jump = max_jump_mult * width_px
    for segment in processed_small:
        seg_scaled = [(x * scale, y * scale) for x, y in segment]
        seg_scaled = sanitize_segment(seg_scaled, img_w, img_h)
        if not seg_scaled:
            continue
        runs += split_large_jumps(seg_scaled, max_jump=max_jump)

    groups = stitch_segments(
        runs, max_join=max_join_mult * width_px, angle_tol=angle_tol
    )
    return img_w, img_h, groups, width_px
