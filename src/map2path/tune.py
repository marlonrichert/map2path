from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from .image_ops import eval_coverage, rasterize_groups


def tune_width_downscaled(
    src_path: Path,
    img_width: int,
    img_height: int,
    groups,
    width_px: float,
    *,
    alpha_thresh: int = 8,
    scale: float = 0.2,
    s_min: float = 0.9,
    s_max: float = 1.1,
    steps: int = 9,
    miss_wt: float = 1.0,
    over_wt: float = 1.0,
) -> Tuple[float, float, float]:
    # Search at lower resolution is fast and resists pixel-level noise.
    image = Image.open(src_path).convert("RGBA")
    w_small = max(1, int(img_width * scale))
    h_small = max(1, int(img_height * scale))
    mask = (
        np.array(
            image.getchannel("A").resize((w_small, h_small), Image.BILINEAR)
        )
        > alpha_thresh
    )
    groups_small = [
        [(x * scale, y * scale) for x, y in segment] for segment in groups
    ]

    best = (1e9, 1.0, 0.0, 0.0)  # score, s, miss, over
    for s in np.linspace(s_min, s_max, steps):
        drawn = rasterize_groups(
            w_small, h_small, groups_small, width_px * s * scale
        )
        miss, over = eval_coverage(mask, drawn)
        score = miss_wt * miss + over_wt * over
        if score < best[0]:
            best = (score, float(s), miss, over)
    return width_px * best[1], best[2], best[3]
