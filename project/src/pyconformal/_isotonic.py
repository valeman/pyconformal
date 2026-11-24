"""Row-wise isotonic regression (pool adjacent violators), compiled when numba is available."""

from __future__ import annotations

import numpy as np


def _pava_rows(X, out):
    M, K = X.shape
    val = np.empty(K)
    wt = np.empty(K)
    for m in range(M):
        n = 0
        for k in range(K):
            val[n] = X[m, k]
            wt[n] = 1.0
            n += 1
            while n > 1 and val[n - 2] > val[n - 1]:
                w = wt[n - 2] + wt[n - 1]
                val[n - 2] = (val[n - 2] * wt[n - 2] + val[n - 1] * wt[n - 1]) / w
                wt[n - 2] = w
                n -= 1
        k = 0
        for b in range(n):
            for _ in range(int(wt[b])):
                out[m, k] = val[b]
                k += 1
    return out


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    _pava_rows = njit(cache=True)(_pava_rows)
except Exception:
    pass


def isotonic_rows(X):
    """Least-squares projection of each row of ``X`` onto non-decreasing vectors."""
    X = np.ascontiguousarray(X, dtype=float)
    return _pava_rows(X, np.empty_like(X))
