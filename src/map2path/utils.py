from __future__ import annotations

import glob
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .constants import IMAGE_EXTENSIONS


def url_safe(text: str) -> str:
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^A-Za-z0-9._-]", "", text)


def expand_inputs(args: Iterable[str]) -> List[Path]:
    # Accept dirs, globs, and specific files so shell patterns just work.
    files: List[str] = []
    for item in args:
        p = Path(item)
        if p.is_dir():
            for suffix in IMAGE_EXTENSIONS:
                files += glob.glob(str(p / f"**/*{suffix}"), recursive=True)
        else:
            matches = glob.glob(item)
            if matches:
                files += matches
            elif p.suffix.lower() in IMAGE_EXTENSIONS and p.exists():
                files.append(str(p))
    files = sorted({str(Path(f)) for f in files})
    return [Path(f) for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS]


def deduplicate_points(
    points: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    # Removing exact repeats avoids degenerate segments after smoothing/resampling.
    out: List[Tuple[float, float]] = []
    previous: Optional[Tuple[float, float]] = None
    for point in points:
        if previous is None or point[0] != previous[0] or point[1] != previous[1]:
            out.append(point)
        previous = point
    return out


def arc_length(points: Sequence[Tuple[float, float]]) -> List[float]:
    # Cumulative length supports uniform re-sampling along the curve.
    acc = [0.0]
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        acc.append(acc[-1] + math.hypot(dx, dy))
    return acc


def resample_by_arclength(
    points: Sequence[Tuple[float, float]],
    count: int,
) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return list(points)
    lengths = arc_length(points)
    if lengths[-1] == 0:
        return list(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    u = np.linspace(0, lengths[-1], count)
    xr = np.interp(u, lengths, xs)
    yr = np.interp(u, lengths, ys)
    return list(zip(xr, yr))


def smooth_moving_average(
    points: Sequence[Tuple[float, float]],
    kernel: int,
) -> List[Tuple[float, float]]:
    # Small box filter removes resampling jaggies without biasing geometry much.
    if not kernel or len(points) < kernel:
        return list(points)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ker = np.ones(kernel) / kernel
    xs = np.convolve(xs, ker, mode="same")
    ys = np.convolve(ys, ker, mode="same")
    return list(zip(xs, ys))


def sanitize_segment(
    segment: Sequence[Tuple[float, float]],
    img_width: int,
    img_height: int,
) -> List[Tuple[float, float]]:
    # SVG consumers can choke on NaNs or wild coords; clamp and drop early.
    out: List[Tuple[float, float]] = []
    for x, y in segment:
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if -1 <= x <= img_width + 1 and -1 <= y <= img_height + 1:
            out.append((float(x), float(y)))
    return out if len(out) >= 2 else []


def split_large_jumps(
    segment: Sequence[Tuple[float, float]],
    max_jump: float,
) -> List[List[Tuple[float, float]]]:
    # Breaking long steps avoids bridging holes created by simplification.
    if not segment:
        return []
    out: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = [segment[0]]
    for i in range(1, len(segment)):
        x0, y0 = segment[i - 1]
        x1, y1 = segment[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist > max_jump:
            if len(current) >= 2:
                out.append(current)
            current = [segment[i]]
        else:
            current.append(segment[i])
    if len(current) >= 2:
        out.append(current)
    return out


def endpoint_direction_vector(
    segment: Sequence[Tuple[float, float]],
    at_start: bool,
    k: int = 5,
) -> Tuple[float, float]:
    # Coarse tangent stabilizes angle checks when endpoints are slightly noisy.
    if len(segment) < 2:
        return (0.0, 0.0)
    if at_start:
        start_point, k_point = segment[0], segment[min(k, len(segment) - 1)]
    else:
        start_point, k_point = segment[max(0, len(segment) - 1 - k)], segment[-1]
    dx, dy = (k_point[0] - start_point[0]), (k_point[1] - start_point[1])
    n = math.hypot(dx, dy) or 1e-9
    return (dx / n, dy / n)


def angle_degrees(
    vector_u: Tuple[float, float],
    vector_v: Tuple[float, float],
) -> float:
    # Clamping mitigates floating errors that otherwise break acos near ±1.
    ux, uy = vector_u
    vx, vy = vector_v
    nu = math.hypot(ux, uy) or 1e-9
    nv = math.hypot(vx, vy) or 1e-9
    c = max(-1.0, min(1.0, (ux * vx + uy * vy) / (nu * nv)))
    return math.degrees(math.acos(c))


def rdp(points: Sequence[Tuple[float, float]], epsilon: float) -> List[
    Tuple[float, float]
]:
    # Ramer–Douglas–Peucker reduces oversampling so later steps scale sanely.
    if len(points) < 3:
        return list(points)

    ax, ay = points[0]
    bx, by = points[-1]
    dx_ab = bx - ax
    dy_ab = by - ay
    len_ab_sq = dx_ab * dx_ab + dy_ab * dy_ab or 1e-9

    dmax_sq = -1.0
    index = 0
    for i in range(1, len(points) - 1):
        px, py = points[i]
        cx = px - ax
        cy = py - ay
        twice_area = dx_ab * cy - dy_ab * cx
        d2 = (twice_area * twice_area) / len_ab_sq
        if d2 > dmax_sq:
            dmax_sq, index = d2, i

    if dmax_sq > epsilon * epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right

    return [points[0], points[-1]]
