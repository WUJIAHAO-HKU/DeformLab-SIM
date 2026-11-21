"""Order-preserving integer encoding of the existing four-connected DFS.

Component seeds remain row-major; neighbours remain up, left, right, down;
the LIFO stack and append-on-discovery rule are unchanged. Equal-sized
components retain the first encountered component. No sorting is introduced.
"""
import numpy as np


def _largest_component(mask):
    height, width = mask.shape
    seeds = np.flatnonzero(mask).tolist()
    remaining = set(seeds)
    best = []
    components = 0
    for p in seeds:
        if p not in remaining:
            continue
        components += 1
        remaining.remove(p)
        stack = [p]
        found = [p]
        while stack:
            i = stack.pop()
            x = i % width
            # Sentinel -1 is never a valid flat index. Horizontal boundaries
            # must not wrap into the adjacent image row.
            for q in (i - width, i - 1 if x else -1,
                      i + 1 if x + 1 < width else -1, i + width):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    found.append(q)
        if len(found) > len(best):
            best = found
    if not best:
        return np.empty((0, 2), dtype=int), components
    indices = np.asarray(best, dtype=int)
    return np.column_stack((indices // width, indices % width)), components
