from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .utils import angle_degrees, endpoint_direction_vector


def stitch_segments(
    runs: List[List[Tuple[float, float]]],
    max_join: float,
    angle_tol: float,
) -> List[List[Tuple[float, float]]]:
    # Greedy stitch with direction gating prefers visually continuous joins.
    if not runs:
        return []

    def polyline_length(segment: Sequence[Tuple[float, float]]) -> float:
        return sum(
            math.hypot(
                segment[i + 1][0] - segment[i][0],
                segment[i + 1][1] - segment[i][1],
            )
            for i in range(len(segment) - 1)
        )

    unused: List[List[Tuple[float, float]]] = [list(seg) for seg in runs]
    stitched: List[List[Tuple[float, float]]] = []
    unused.sort(key=lambda s: -polyline_length(s))  # prioritize long backbones

    while unused:
        path = unused.pop(0)
        extended = True
        while extended and unused:
            extended = False
            best_index = -1
            best_to_front = False
            best_flip = False
            best_dist2 = 1e18

            head, tail = path[0], path[-1]
            head_dir = endpoint_direction_vector(path, at_start=True)
            tail_dir = endpoint_direction_vector(path, at_start=False)

            for i, candidate in enumerate(unused):
                start_pt, end_pt = candidate[0], candidate[-1]

                # append to tail
                for end_point, flip_flag in (start_pt, False), (end_pt, True):
                    dx = end_point[0] - tail[0]
                    dy = end_point[1] - tail[1]
                    d2 = dx * dx + dy * dy
                    if (
                        d2 <= max_join * max_join
                        and angle_degrees(tail_dir, (dx, dy)) <= angle_tol
                        and d2 < best_dist2
                    ):
                        best_dist2 = d2
                        best_index = i
                        best_to_front = False
                        best_flip = flip_flag

                # prepend to head
                for end_point, flip_flag in (end_pt, False), (start_pt, True):
                    dx = head[0] - end_point[0]
                    dy = head[1] - end_point[1]
                    d2 = dx * dx + dy * dy
                    if d2 <= max_join * max_join:
                        cand_dir = endpoint_direction_vector(
                            candidate if not flip_flag else list(reversed(candidate)),
                            at_start=not flip_flag,
                        )
                        if (
                            angle_degrees(cand_dir, (dx, dy)) <= angle_tol
                            and angle_degrees(head_dir, (-dx, -dy)) <= angle_tol
                            and d2 < best_dist2
                        ):
                            best_dist2 = d2
                            best_index = i
                            best_to_front = True
                            best_flip = flip_flag

            if best_index >= 0:
                segment = unused.pop(best_index)
                if best_flip:
                    segment = list(reversed(segment))
                path = segment + path if best_to_front else path + segment
                extended = True

        stitched.append(path)
    return stitched


def force_single_path(
    groups: List[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    # Simple nearest joining builds one monopath for devices that require it.
    if not groups:
        return []
    unused: List[List[Tuple[float, float]]] = [list(seg) for seg in groups]
    path = unused.pop(0)
    while unused:
        head, tail = path[0], path[-1]
        best = (1e18, None, False, False)  # dist2, idx, flip, prepend
        for i, segment in enumerate(unused):
            d_tail0 = (segment[0][0] - tail[0]) ** 2 + (segment[0][1] - tail[1]) ** 2
            d_tail1 = (segment[-1][0] - tail[0]) ** 2 + (
                segment[-1][1] - tail[1]
            ) ** 2
            if d_tail0 < best[0]:
                best = (d_tail0, i, False, False)
            if d_tail1 < best[0]:
                best = (d_tail1, i, True, False)
            d_head0 = (head[0] - segment[0][0]) ** 2 + (head[1] - segment[0][1]) ** 2
            d_head1 = (head[0] - segment[-1][0]) ** 2 + (
                head[1] - segment[-1][1]
            ) ** 2
            if d_head0 < best[0]:
                best = (d_head0, i, False, True)
            if d_head1 < best[0]:
                best = (d_head1, i, True, True)
        _, idx, flip, prepend = best
        chosen = unused.pop(idx)  # type: ignore[arg-type]
        if flip:
            chosen = list(reversed(chosen))
        path = chosen + path if prepend else path + chosen
    return path
