from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .constants import NEIGHBOR_OFFSETS


def iter_neighbors(
    point_row_col: Tuple[int, int],
    foreground: np.ndarray,
) -> Iterable[Tuple[int, int]]:
    # 8-connected neighborhood avoids topological breaks at diagonals.
    row, col = point_row_col
    height, width = foreground.shape
    for d_row, d_col in NEIGHBOR_OFFSETS:
        rr, cc = row + d_row, col + d_col
        if 0 <= rr < height and 0 <= cc < width and foreground[rr, cc]:
            yield (rr, cc)


def degree_map(foreground: np.ndarray) -> Dict[Tuple[int, int], int]:
    indices = np.argwhere(foreground)
    result: Dict[Tuple[int, int], int] = {}
    for row, col in map(tuple, indices):
        result[(row, col)] = sum(
            1 for _ in iter_neighbors((row, col), foreground)
        )
    return result


def prune_spurs(
    foreground: np.ndarray,
    max_len: int = 15,
    max_iter: int = 20,
) -> np.ndarray:
    # Clipping hairs prevents the stitcher from chasing dead‑ends out of bounds.
    fg = foreground.copy()
    for _ in range(max_iter):
        changed = False
        deg = degree_map(fg)
        endpoints = [p for p, k in deg.items() if k == 1]

        for start in endpoints:
            if not fg[start]:
                continue
            previous: Optional[Tuple[int, int]] = None
            current: Tuple[int, int] = start
            path: List[Tuple[int, int]] = [start]
            steps = 0

            while True:
                neighbor_points = [
                    q for q in iter_neighbors(current, fg) if q != previous
                ]
                if len(neighbor_points) == 0 or len(neighbor_points) >= 2:
                    break
                nxt = neighbor_points[0]
                path.append(nxt)
                previous, current = current, nxt
                steps += 1
                if steps > max_len:
                    break

            if steps <= max_len:
                for r, c in path:
                    fg[r, c] = False
                changed = True

        if not changed:
            break
    return fg


def trace_skeleton(foreground: np.ndarray) -> List[List[Tuple[float, float]]]:
    # Greedy edge walk yields local polylines; crossings will be regularized later.
    points = np.argwhere(foreground)
    if len(points) == 0:
        return []

    deg = degree_map(foreground)
    visited: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    lines: List[List[Tuple[int, int]]] = []

    def walk(
        start: Tuple[int, int],
        prev: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        path = [start]
        cur, pre = start, prev
        while True:
            neighbor_points = [q for q in iter_neighbors(cur, foreground)]
            if pre is not None and pre in neighbor_points:
                neighbor_points.remove(pre)

            next_candidates = [
                q for q in neighbor_points if tuple(sorted((cur, q))) not in visited
            ]
            if not next_candidates:
                return path

            next_candidates.sort(
                key=lambda q: 0 if deg.get(q, 0) == 2 else 1
            )
            q = next_candidates[0]
            visited.add(tuple(sorted((cur, q))))
            pre, cur = cur, q
            path.append(cur)
            if deg.get(cur, 0) != 2:
                return path

    for r, c in map(tuple, points):
        for q in iter_neighbors((r, c), foreground):
            edge = tuple(sorted(((r, c), q)))
            if edge not in visited:
                visited.add(edge)
                lines.append([(r, c)] + walk(q, prev=(r, c)))

    return [[(c * 1.0, r * 1.0) for r, c in seg] for seg in lines]


def junction_pixels(
    foreground: np.ndarray,
    min_deg: int = 4,
) -> Set[Tuple[int, int]]:
    # Degree≥4 is a robust proxy for real crossings on 8‑connected skeletons.
    deg = degree_map(foreground)
    return {p for p, k in deg.items() if k >= min_deg}
