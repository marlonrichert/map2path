from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .pipeline import process_one
from .svg import save_centerline_svg, save_overlay_svg
from .tune import tune_width_downscaled
from .utils import url_safe


@dataclass(frozen=True)
class AppConfig:
    out_dir: str = "svg_centerlines"
    max_dim: int = 1600
    spur: int = 15
    eps: float = 1.0
    resample_per_px: float = 0.30
    resample_min: int = 50
    resample_max: int = 900
    smooth_k: int = 3
    alpha_thresh: int = 8
    max_jump_mult: float = 1.75
    max_join_mult: float = 0.45
    angle_tol: float = 10.0
    force_single_path: bool = False
    overlay: bool = False
    auto_width: bool = False
    width_range: tuple[float, float] = (0.9, 1.1)
    width_steps: int = 9
    miss_weight: float = 1.0
    over_weight: float = 1.0


class CenterlineApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.out_dir_path = Path(config.out_dir)
        self.out_dir_path.mkdir(parents=True, exist_ok=True)

    def process_path(self, image_path: Path):
        return process_one(
            image_path,
            max_dim=self.config.max_dim,
            spur=self.config.spur,
            eps=self.config.eps,
            resample_ppx=self.config.resample_per_px,
            resample_min=self.config.resample_min,
            resample_max=self.config.resample_max,
            smooth_k=self.config.smooth_k,
            alpha_thresh=self.config.alpha_thresh,
            max_jump_mult=self.config.max_jump_mult,
            max_join_mult=self.config.max_join_mult,
            angle_tol=self.config.angle_tol,
        )

    def maybe_tune_width(
        self,
        image_path: Path,
        img_width: int,
        img_height: int,
        groups,
        width_px: float,
    ) -> float:
        if not self.config.auto_width:
            return width_px
        smin, smax = self.config.width_range
        tuned, _, _ = tune_width_downscaled(
            image_path,
            img_width,
            img_height,
            groups,
            width_px,
            alpha_thresh=self.config.alpha_thresh,
            scale=0.2,
            s_min=smin,
            s_max=smax,
            steps=self.config.width_steps,
            miss_wt=self.config.miss_weight,
            over_wt=self.config.over_weight,
        )
        return tuned

    def save_outputs(
        self,
        image_path: Path,
        img_width: int,
        img_height: int,
        groups,
        width_px: float,
    ) -> List[Path]:
        stem = url_safe(image_path.stem)
        centerline_path = self.out_dir_path / f"{stem}_centerline.svg"
        save_centerline_svg(
            centerline_path,
            img_width,
            img_height,
            groups,
            width_px,
            self.config.force_single_path,
        )
        written = [centerline_path]
        print(
            f"[ok] {image_path} -> {centerline_path} "
            f"({'single' if self.config.force_single_path else 'multi'} "
            f"path, width≈{width_px:.2f}px)"
        )
        if self.config.overlay:
            overlay_path = self.out_dir_path / f"{stem}_overlay.svg"
            save_overlay_svg(
                image_path,
                overlay_path,
                img_width,
                img_height,
                groups,
                width_px,
                self.config.force_single_path,
            )
            written.append(overlay_path)
            print(f"[ok] overlay -> {overlay_path}")
        return written


def extract_centerlines(
    inputs: Iterable[str | Path],
    *,
    force_single_path: bool = False,
    overlay: bool = False,
    out_dir: str = "svg_centerlines",
    **kwargs,
) -> List[Path]:
    """
    Programmatic batch API.
    Keeps CLI defaults; kwargs map to AppConfig fields where relevant.
    """
    cfg = AppConfig(
        out_dir=out_dir,
        force_single_path=force_single_path,
        overlay=overlay,
        **{
            k: v
            for k, v in kwargs.items()
            if k in AppConfig.__annotations__
            and k not in {"out_dir", "overlay", "force_single_path"}
        },
    )
    app = CenterlineApp(cfg)
    from .utils import expand_inputs  # local import to keep top-level light

    paths = expand_inputs(list(inputs))
    written: List[Path] = []
    for p in paths:
        img_w, img_h, groups, width_px = app.process_path(p)
        if not groups:
            continue
        width_px = app.maybe_tune_width(p, img_w, img_h, groups, width_px)
        written += app.save_outputs(p, img_w, img_h, groups, width_px)
    return written
