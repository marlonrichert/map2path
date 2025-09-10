from __future__ import annotations

from typing import List, Sequence, Set, Tuple

import numpy as np


def regularize_crossings(
    segments: List[List[Tuple[float, float]]],
    junctions: Set[Tuple[int, int]],
    steps: int = 6,
) -> List[List[Tuple[float, float]]]:
    # Straightening opposite arms at crossings removes post-smooth wobble.
    index: dict[Tuple[int, int], list[Tuple[int, int]]] = {}
    for seg_idx, seg in enumerate(segments):
        for i, (x, y) in enumerate(seg):
            rc = (int(round(y)), int(round(x)))
            if rc in junctions:
                index.setdefault(rc, []).append((seg_idx, i))

    for rc, occurrences in index.items():
        if len(occurrences) < 4:
            continue
        r, c = rc
        center = np.array([c * 1.0, r * 1.0])

        arms: list[tuple[int, int, np.ndarray]] = []
        for seg_idx, i in occurrences:
            seg = segments[seg_idx]
            if i + 1 < len(seg):
                direction = np.array(seg[i + 1]) - np.array(seg[i])
            elif i - 1 >= 0:
                direction = np.array(seg[i]) - np.array(seg[i - 1])
            else:
                continue
            n = np.linalg.norm(direction) or 1e-9
            arms.append((seg_idx, i, direction / n))

        if len(arms) < 4:
            continue

        used: set[int] = set()
        pairs: list[
            tuple[tuple[int, int, np.ndarray], tuple[int, int, np.ndarray]]
        ] = []
        for _ in range(2):
            best = None
            best_val = -1.0
            for a in range(len(arms)):
                if a in used:
                    continue
                for b in range(a + 1, len(arms)):
                    if b in used:
                        continue
                    dot = float(np.dot(arms[a][2], arms[b][2]))
                    val = -dot
                    if val > best_val:
                        best_val = val
                        best = (a, b)
            if not best:
                break
            a, b = best
            used.add(a)
            used.add(b)
            pairs.append((arms[a], arms[b]))

        for (seg_a, idx_a, dir_a), (seg_b, idx_b, dir_b) in pairs:
            if float(np.dot(dir_a, dir_b)) > 0:
                dir_b = -dir_b
            axis_vec = dir_a - dir_b
            n = np.linalg.norm(axis_vec) or 1e-9
            axis_vec = axis_vec / n
            for (seg_idx, i0) in [(seg_a, idx_a), (seg_b, idx_b)]:
                seg = segments[seg_idx]
                lo = max(0, i0 - steps)
                hi = min(len(seg), i0 + steps + 1)
                for j in range(lo, hi):
                    p = np.array(seg[j])
                    t = float(np.dot(p - center, axis_vec))
                    seg[j] = tuple((center + t * axis_vec).tolist())
    return segments
