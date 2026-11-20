"""Binary search over a sorted array for many values at once.

Same results as ``numpy.searchsorted``; with numba installed and enough
values, the searches run on all cores.
"""

from __future__ import annotations

import numpy as np

_PARALLEL_MIN = 20_000

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _search_nb(a, v, right, out):
        n = a.shape[0]
        for t in prange(v.shape[0]):
            x = v[t]
            lo = 0
            hi = n
            if right:
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if a[mid] <= x:
                        lo = mid + 1
                    else:
                        hi = mid
            else:
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if a[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
            out[t] = lo

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def search(a, v, side="left"):
    """Insertion points of ``v`` in sorted ``a`` (both float64, 1-D)."""
    if HAVE_NUMBA and v.shape[0] >= _PARALLEL_MIN:
        out = np.empty(v.shape[0], dtype=np.int64)
        _search_nb(a, np.ascontiguousarray(v), side == "right", out)
        return out
    return np.searchsorted(a, v, side=side)
