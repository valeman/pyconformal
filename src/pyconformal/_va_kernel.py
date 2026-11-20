"""Linear-time kernel for inductive Venn-Abers predictors.

Geometry. Sort the calibration scores and group equal scores. Let
``X[j]`` and ``Y[j]`` be the number of calibration points and the number of
positive labels in the first ``j`` groups (``X[0] = Y[0] = 0``). The isotonic
regression of the labels is the slope of the greatest convex minorant of
the points ``(X[j], Y[j])`` (Barlow et al., 1972).

Adding one test point with label ``dy`` (0 or 1) to group ``g`` shifts every
point to the right of group ``g`` by ``(1, dy)``. The isotonic fit at the
test point is then the slope of the lower common tangent ("bridge") between

* the lower convex hull of the unshifted points ``0..g``, and
* the lower convex hull of the shifted points ``g+1..k``.

Both families of hulls are stored once as pointer chains (``prv`` and
``nxt``), built by a Graham scan in linear time. Each bridge is found by
walking the two chains from their inner ends.

All comparisons use integer cross products, so there is no rounding in the
geometry; the only floating-point operation is the final division.
"""

from __future__ import annotations


def _bridge_py(X, Y, dy, prv, nxt, num, den):
    """Fill ``num[g] / den[g]`` = isotonic fit at group ``g`` after adding one
    point with label ``dy`` to that group. ``X`` and ``Y`` have length k + 1.
    ``prv``, ``nxt``, ``num`` and ``den`` are work/output buffers of length k.
    """
    k = len(X) - 1

    # prv[g]: previous vertex on the lower hull of points 0..g.
    prv[0] = -1
    for g in range(1, k):
        c = g - 1
        while prv[c] >= 0:
            p = prv[c]
            if (X[c] - X[p]) * (Y[g] - Y[c]) - (Y[c] - Y[p]) * (X[g] - X[c]) <= 0:
                c = p
            else:
                break
        prv[g] = c

    # nxt[b]: next vertex on the lower hull of points b+1..k (indexed by b).
    nxt[k - 1] = -1
    for b in range(k - 2, -1, -1):
        c = b + 1
        while nxt[c] >= 0:
            q = nxt[c]
            if (X[c + 1] - X[b + 1]) * (Y[q + 1] - Y[c + 1]) - (
                Y[c + 1] - Y[b + 1]
            ) * (X[q + 1] - X[c + 1]) <= 0:
                c = q
            else:
                break
        nxt[b] = c

    for g in range(k):
        a = g
        b = g
        ax = X[a]
        ay = Y[a]
        bx = X[b + 1] + 1
        by = Y[b + 1] + dy
        while True:
            moved = False
            # Right end: minimise the slope seen from the left end.
            while nxt[b] >= 0:
                q = nxt[b]
                qx = X[q + 1] + 1
                qy = Y[q + 1] + dy
                if (qy - ay) * (bx - ax) < (by - ay) * (qx - ax):
                    b = q
                    bx = qx
                    by = qy
                    moved = True
                else:
                    break
            # Left end: maximise the slope seen from the right end.
            while prv[a] >= 0:
                p = prv[a]
                px = X[p]
                py = Y[p]
                if (by - py) * (bx - ax) > (by - ay) * (bx - px):
                    a = p
                    ax = px
                    ay = py
                    moved = True
                else:
                    break
            if not moved:
                break
        num[g] = by - ay
        den[g] = bx - ax


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    _bridge_nb = njit(cache=True, nogil=True)(_bridge_py)
    HAVE_NUMBA = True
except Exception:  # numba missing or unusable on this Python
    _bridge_nb = None
    HAVE_NUMBA = False


def bridge_slopes(X, Y, dy, use_numba=None):
    """Return the float array of fits for label ``dy`` at every group.

    ``X`` and ``Y`` are int64 NumPy arrays of cumulative counts.
    """
    import numpy as np

    k = len(X) - 1
    if use_numba is None:
        use_numba = HAVE_NUMBA
    if use_numba:
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available.")
        prv = np.empty(k, dtype=np.int64)
        nxt = np.empty(k, dtype=np.int64)
        num = np.empty(k, dtype=np.int64)
        den = np.empty(k, dtype=np.int64)
        _bridge_nb(X, Y, dy, prv, nxt, num, den)
        return num / den
    prv = [0] * k
    nxt = [0] * k
    num = [0] * k
    den = [0] * k
    _bridge_py(X.tolist(), Y.tolist(), int(dy), prv, nxt, num, den)
    return np.asarray(num, dtype=float) / np.asarray(den, dtype=float)
