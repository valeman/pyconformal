"""Kernel for adaptive conformal inference over a whole stream.

The calibration set changes at every step and the requested rank changes
with the working alpha, so each step needs "the k-th smallest of a changing
set". All scores are known when a stream is replayed, so they are ranked
once and kept in a Fenwick (binary indexed) tree of counts. Insert, delete
and k-th-smallest are then O(log N), and a stream of T steps costs
O((n + T) log(n + T)) in total.
"""

from __future__ import annotations

import math

# Guards against (n + 1) * (1 - alpha) landing a hair above an integer in
# floating point (for example 19.000000000000004 instead of 19).
_RANK_TOL = 1e-9


def _aci_py(vals, rank, n0, alpha, gamma, alpha_t, window, tree, q, err, alphas):
    """Replay a stream.

    vals   : all scores sorted ascending (calibration and stream together)
    rank   : rank[i] = position in ``vals`` of the i-th score in arrival
             order (first the n0 calibration scores, then the stream)
    window : 0 for an expanding calibration set, else its maximum size
    tree   : zeroed Fenwick tree of length N + 1
    q, err, alphas : outputs of length T
    Returns the working alpha after the last step.
    """
    N = len(vals)
    T = N - n0
    top = 1
    while top * 2 <= N:
        top *= 2

    head = 0  # arrival index of the oldest score still in the set
    if window > 0 and n0 > window:
        head = n0 - window
    n = 0
    for i in range(head, n0):
        j = rank[i] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1

    for t in range(T):
        alphas[t] = alpha_t
        s = vals[rank[n0 + t]]
        if alpha_t <= 0.0:
            thr = math.inf
        else:
            k = math.ceil((n + 1) * (1.0 - alpha_t) - _RANK_TOL)
            if k > n:
                thr = math.inf
            elif k < 1:
                thr = -math.inf
            else:
                pos = 0
                step = top
                while step > 0:
                    nxt = pos + step
                    if nxt <= N and tree[nxt] < k:
                        pos = nxt
                        k -= tree[nxt]
                    step //= 2
                thr = vals[pos]
        q[t] = thr
        miss = s > thr
        err[t] = miss
        alpha_t = alpha_t + gamma * (alpha - (1.0 if miss else 0.0))

        j = rank[n0 + t] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
        if window > 0 and n > window:
            j = rank[head] + 1
            while j <= N:
                tree[j] -= 1
                j += j & (-j)
            head += 1
            n -= 1
    return alpha_t


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    _aci_nb = njit(cache=True, nogil=True)(_aci_py)
    HAVE_NUMBA = True
except Exception:
    _aci_nb = None
    HAVE_NUMBA = False


def aci_replay(cal, stream, alpha, gamma, alpha_t, window, use_numba=None):
    """Run ACI over ``stream`` starting from calibration scores ``cal``.

    Returns (thresholds, errors, alphas, final_alpha_t).
    """
    import numpy as np

    n0 = cal.shape[0]
    T = stream.shape[0]
    allv = np.concatenate([cal, stream])
    order = np.argsort(allv, kind="stable")
    rank = np.empty(n0 + T, dtype=np.int64)
    rank[order] = np.arange(n0 + T)
    vals = allv[order]
    if use_numba is None:
        use_numba = HAVE_NUMBA
    if use_numba:
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available.")
        tree = np.zeros(n0 + T + 1, dtype=np.int64)
        q = np.empty(T)
        err = np.empty(T, dtype=np.bool_)
        alphas = np.empty(T)
        final = _aci_nb(vals, rank, n0, alpha, gamma, alpha_t, window, tree, q, err, alphas)
        return q, err, alphas, float(final)
    tree = [0] * (n0 + T + 1)
    q = [0.0] * T
    err = [False] * T
    alphas = [0.0] * T
    final = _aci_py(
        vals.tolist(), rank.tolist(), n0, alpha, gamma, alpha_t, window, tree, q, err, alphas
    )
    return np.asarray(q), np.asarray(err, dtype=bool), np.asarray(alphas), float(final)
