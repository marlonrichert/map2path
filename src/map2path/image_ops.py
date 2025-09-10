from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_mask(path: Path, max_dim: int, alpha_thresh: int):
    # Downscaling via alpha channel preserves stroke topology at lower cost.
    image = Image.open(path).convert("RGBA")
    width_px, height_px = image.size
    if max(width_px, height_px) > max_dim:
        scale = max(width_px, height_px) / float(max_dim)
        scaled_width = int(round(width_px / scale))
        scaled_height = int(round(height_px / scale))
        alpha = image.getchannel("A").resize(
            (scaled_width, scaled_height), Image.BILINEAR
        )
    else:
        scaled_width, scaled_height = width_px, height_px
        alpha = image.getchannel("A")
    mask = np.array(alpha) > alpha_thresh
    return mask, (width_px, height_px), (scaled_width, scaled_height)


def rasterize_groups(
    img_width: int,
    img_height: int,
    groups,
    stroke_width: float,
) -> np.ndarray:
    # Low‑res raster estimate is much faster for width tuning than full‑res vector.
    image = Image.new("L", (img_width, img_height), 0)
    drawer = ImageDraw.Draw(image)
    w = max(1, int(round(stroke_width)))
    for segment in groups:
        if len(segment) >= 2:
            drawer.line(segment, fill=255, width=w, joint="curve")
    return np.array(image) > 0


def eval_coverage(mask: np.ndarray, drawn: np.ndarray) -> Tuple[float, float]:
    # Score balances missing painted pixels vs. overpaint outside the alpha.
    misses = mask & (~drawn)
    overs = drawn & (~mask)
    fg = int(mask.sum())
    path = int(drawn.sum())
    miss_pct = (misses.sum() / fg * 100.0) if fg else 0.0
    over_pct = (overs.sum() / path * 100.0) if path else 0.0
    return float(miss_pct), float(over_pct)
