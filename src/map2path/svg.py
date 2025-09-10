from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Sequence, Tuple

from .stitch import force_single_path
from .utils import deduplicate_points


def stitched_groups_to_path_d(
    groups: Sequence[Sequence[Tuple[float, float]]],
) -> str:
    # Single path renders faster and reduces file size compared to many <polyline>s.
    parts: List[str] = []
    for segment in groups:
        segment = deduplicate_points(segment)
        if not segment:
            continue
        parts.append(f"M {segment[0][0]:.2f} {segment[0][1]:.2f}")
        parts += [f"L {x:.2f} {y:.2f}" for x, y in segment[1:]]
    return " ".join(parts)


def single_path_d(points: Sequence[Tuple[float, float]]) -> str:
    deduped = deduplicate_points(points)
    if not deduped:
        return ""
    return " ".join(
        [f"M {deduped[0][0]:.2f} {deduped[0][1]:.2f}"]
        + [f"L {x:.2f} {y:.2f}" for x, y in deduped[1:]]
    )


def save_centerline_svg(
    out_path: Path,
    img_width: int,
    img_height: int,
    groups,
    width_px: float,
    force_single: bool,
) -> None:
    if force_single:
        pts = force_single_path(groups)
        d_attr = single_path_d(pts)
    else:
        d_attr = stitched_groups_to_path_d(groups)
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{img_width}' "
        f"height='{img_height}' viewBox='0 0 {img_width} {img_height}'>\n"
        f"<path d=\"{d_attr}\" fill='none' stroke='#000000' "
        f"stroke-opacity='0.5' stroke-width='{width_px:.2f}' "
        f"stroke-linecap='round' stroke-linejoin='round' "
        f"stroke-miterlimit='1'/>\n"
        f"</svg>\n"
    )
    out_path.write_text(svg, encoding="utf-8")


def save_overlay_svg(
    src_path: Path,
    out_path: Path,
    img_width: int,
    img_height: int,
    groups,
    width_px: float,
    force_single: bool,
) -> None:
    # Embedding the source image as data URI avoids sidecar asset management.
    b64 = base64.b64encode(src_path.read_bytes()).decode("ascii")
    img_tag = (
        f"<image href='data:image/png;base64,{b64}' x='0' y='0' "
        f"width='{img_width}' height='{img_height}'/>"
    )
    if force_single:
        pts = force_single_path(groups)
        d_attr = single_path_d(pts)
        segs = [deduplicate_points(pts)]
    else:
        d_attr = stitched_groups_to_path_d(groups)
        segs = [deduplicate_points(s) for s in groups if s]

    def polyline(segment: Sequence[Tuple[float, float]]) -> str:
        pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
        return (
            f"<polyline points='{pts_str}' fill='none' stroke='#FFFFFF' "
            f"stroke-opacity='0.5' stroke-width='1'/>"
        )

    rects = []
    for segment in segs:
        for x, y in segment:
            rects.append(
                f"<rect x='{x-1.5:.2f}' y='{y-1.5:.2f}' width='3' height='3' "
                f"fill='#FFFFFF' fill-opacity='0.5'/>"
            )
    lines = [polyline(segment) for segment in segs if len(segment) >= 2]
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{img_width}' "
        f"height='{img_height}' viewBox='0 0 {img_width} {img_height}'>\n"
        f"{img_tag}\n"
        f"<path d=\"{d_attr}\" fill='none' stroke='#000000' "
        f"stroke-opacity='0.5' stroke-width='{width_px:.2f}' "
        f"stroke-linecap='round' stroke-linejoin='round' "
        f"stroke-miterlimit='1'/>\n"
        f"{''.join(lines)}\n"
        f"{''.join(rects)}\n"
        f"</svg>\n"
    )
    out_path.write_text(svg, encoding="utf-8")
