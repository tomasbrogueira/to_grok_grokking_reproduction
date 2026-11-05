"""Grokking times t1 and t2 (Theorem 4.2) and the curve time grid."""
import numpy as np


def curve_times(tmax=5_000_000):
    """Decades, plus 1.5e k and 5e k from 1e3 on (U7)."""
    ts = {0} | {10**k for k in range(7)}
    ts |= {int(a * 10**k) for a in (1.5, 5) for k in range(3, 7)}
    return sorted(t for t in ts if t <= tmax)


def search_grid(tmax=10**9, points=6000):
    g = np.unique(np.round(np.geomspace(1, tmax, points)).astype(np.int64))
    return np.concatenate([[0], g])


def _bisect(f, lo, hi, threshold):
    """Largest integer in [lo, hi) with f >= threshold, given f(lo) >= threshold > f(hi)."""
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) >= threshold:
            lo = mid
        else:
            hi = mid
    return lo


def last_above(values, grid, threshold, f):
    """t1 = max{t : f(t) >= threshold}. None if f starts below and stays below on the grid."""
    above = np.where(values >= threshold)[0]
    if len(above) == 0:
        return None
    i = above[-1]
    if i == len(grid) - 1:
        return None  # never drops below within the grid
    return int(_bisect(f, int(grid[i]), int(grid[i + 1]), threshold))


def first_at_or_below(values, grid, threshold, f):
    """t2 = min{t : f(t) <= threshold}. None if not reached on the grid."""
    below = np.where(values <= threshold)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return int(grid[0])
    lo, hi = int(grid[i - 1]), int(grid[i])  # f(lo) > threshold >= f(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid) <= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def last_above_sequence(values, threshold):
    idx = [i for i, v in enumerate(values) if v >= threshold]
    return idx[-1] if idx else None


def first_below_sequence(values, threshold):
    for i, v in enumerate(values):
        if v <= threshold:
            return i
    return None
