"""Kernel for one round of W-TQA: weights and weighted quantile for every
target, in one pass and without building the (targets x units) weight
matrix. With numba the targets are handled in parallel."""

from __future__ import annotations

import math

import numpy as np


def _round_py(A, B, inv, order, ss, level_in, finite, c, tol, q, fallback):
    """``A`` (n, d) and ``B`` (M, d): centred running means of the units and
    the targets; ``inv`` (M, d): one over the squared scale of each feature
    for each target; ``order``: indices of the units with a score, sorted
    by score; ``ss``: those scores, sorted; ``c = 1 / (2 d h^2)``."""
    M, d = B.shape
    n = order.shape[0]
    for m in prange(M):
        kap = np.empty(n)
        total = 0.0
        for i in range(n):
            k = order[i]
            d2 = 0.0
            for j in range(d):
                z = A[k, j] - B[m, j]
                d2 += z * z * inv[m, j]
            kap[i] = math.exp(-d2 * c)
            total += kap[i]
        total += 1.0
        w_tgt = 1.0 / total
        a = level_in[m]
        fb = a < w_tgt
        fallback[m] = fb
        level = a
        if finite and fb:
            level = w_tgt
        if level >= 1.0:
            q[m] = -np.inf
            continue
        if finite and fb:
            q[m] = ss[n - 1]
            continue
        need = (1.0 - level) * total - tol * total
        cum = 0.0
        out = np.inf
        for i in range(n):
            cum += kap[i]
            if cum >= need:
                out = ss[i]
                break
        q[m] = out


prange = range
HAVE_NUMBA = False
round_thresholds = _round_py
try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange  # noqa: F811

    round_thresholds = njit(parallel=True, cache=True)(_round_py)
    HAVE_NUMBA = True
except Exception:  # numba missing or unusable on this Python
    prange = range
