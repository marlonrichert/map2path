from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import AppConfig, CenterlineApp
from .utils import expand_inputs


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Centerline extraction with safe stitching and overlay."
    )
    ap.add_argument("inputs", nargs="+", help="Image files, folders, or globs")
    ap.add_argument("--out-dir", default="svg_centerlines")
    ap.add_argument("--max-dim", type=int, default=1600)
    ap.add_argument("--spur", type=int, default=15)
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--resample-per-px", type=float, default=0.30)
    ap.add_argument("--resample-min", type=int, default=50)
    ap.add_argument("--resample-max", type=int, default=900)
    ap.add_argument("--smooth-k", type=int, default=3)
    ap.add_argument("--alpha-thresh", type=int, default=8)
    ap.add_argument(
        "--max-jump-mult",
        type=float,
        default=1.75,
        help="Split if step exceeds this × width.",
    )
    ap.add_argument(
        "--max-join-mult",
        type=float,
        default=0.45,
        help="Join only if gap ≤ this × width.",
    )
    ap.add_argument(
        "--angle-tol",
        type=float,
        default=10.0,
        help="Max endpoint tangent mismatch (deg).",
    )
    ap.add_argument(
        "--force-single-path",
        action="store_true",
        help="Insert straight joins to make one path.",
    )
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="Write <stem>_overlay.svg",
    )
    ap.add_argument(
        "--auto-width",
        action="store_true",
        help="Downscaled search for width scale that balances miss/over.",
    )
    ap.add_argument("--width-range", type=float, nargs=2, default=(0.9, 1.1))
    ap.add_argument("--width-steps", type=int, default=9)
    ap.add_argument("--miss-weight", type=float, default=1.0)
    ap.add_argument("--over-weight", type=float, default=1.0)
    return ap


def to_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        out_dir=args.out_dir,
        max_dim=args.max_dim,
        spur=args.spur,
        eps=args.eps,
        resample_per_px=args.resample_per_px,
        resample_min=args.resample_min,
        resample_max=args.resample_max,
        smooth_k=args.smooth_k,
        alpha_thresh=args.alpha_thresh,
        max_jump_mult=args.max_jump_mult,
        max_join_mult=args.max_join_mult,
        angle_tol=args.angle_tol,
        force_single_path=args.force_single_path,
        overlay=args.overlay,
        auto_width=args.auto_width,
        width_range=tuple(args.width_range),
        width_steps=args.width_steps,
        miss_weight=args.miss_weight,
        over_weight=args.over_weight,
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    inputs = expand_inputs(args.inputs)
    if not inputs:
        print("No images found.", file=sys.stderr)
        return 1

    app = CenterlineApp(to_config(args))

    for image_path in inputs:
        try:
            img_w, img_h, groups, width_px = app.process_path(image_path)
            if not groups:
                print(f"[skip] {image_path} -> no centerlines")
                continue
            width_px = app.maybe_tune_width(
                image_path, img_w, img_h, groups, width_px
            )
            app.save_outputs(image_path, img_w, img_h, groups, width_px)
        except Exception as exc:
            print(
                f"[err] {image_path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return 0
