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
    return _bridge_vec(X, Y, int(dy))


def _hulls_py(X, Y):
    """The two pointer chains of ``_bridge_py``, built with plain lists."""
    k = len(X) - 1
    prv = [0] * k
    nxt = [0] * k
    prv[0] = -1
    for g in range(1, k):
        c = g - 1
        xg = X[g]
        yg = Y[g]
        while prv[c] >= 0:
            p = prv[c]
            if (X[c] - X[p]) * (yg - Y[c]) - (Y[c] - Y[p]) * (xg - X[c]) <= 0:
                c = p
            else:
                break
        prv[g] = c
    nxt[k - 1] = -1
    for b in range(k - 2, -1, -1):
        c = b + 1
        xb = X[b + 1]
        yb = Y[b + 1]
        while nxt[c] >= 0:
            q = nxt[c]
            if (X[c + 1] - xb) * (Y[q + 1] - Y[c + 1]) - (Y[c + 1] - yb) * (X[q + 1] - X[c + 1]) <= 0:
                c = q
            else:
                break
        nxt[b] = c
    return prv, nxt


_HULLS = {"key": None, "value": None}


def _release_hulls():
    _HULLS["key"] = None
    _HULLS["value"] = None


def _bridge_vec(X, Y, dy):
    """The walk of ``_bridge_py`` without numba: every group takes its steps
    at the same time as NumPy array operations, so Python only loops over
    rounds, not over groups. Same moves, same result."""
    import numpy as np

    k = len(X) - 1
    key = (id(X), id(Y), k)
    if _HULLS["key"] != key:  # label 0 and label 1 share the same hulls
        prv, nxt = _hulls_py(X.tolist(), Y.tolist())
        _HULLS["key"] = key
        _HULLS["value"] = (np.asarray(prv, dtype=np.int64), np.asarray(nxt, dtype=np.int64), X, Y)
    prv, nxt = _HULLS["value"][:2]
    a = np.arange(k)
    b = a.copy()
    ax = X[:-1].copy()
    ay = Y[:-1].copy()
    bx = X[1:] + 1
    by = Y[1:] + dy
    act = a.copy()
    while act.size:
        moved = np.zeros(act.size, dtype=bool)
        # right ends: minimise the slope seen from the left end
        idx = act
        loc = np.arange(act.size)
        while idx.size:
            q = nxt[b[idx]]
            ok = q >= 0
            qs = np.where(ok, q, 0)
            qx = X[qs + 1] + 1
            qy = Y[qs + 1] + dy
            adv = ok & ((qy - ay[idx]) * (bx[idx] - ax[idx]) < (by[idx] - ay[idx]) * (qx - ax[idx]))
            idx = idx[adv]
            loc = loc[adv]
            b[idx] = q[adv]
            bx[idx] = qx[adv]
            by[idx] = qy[adv]
            moved[loc] = True
        # left ends: maximise the slope seen from the right end
        idx = act
        loc = np.arange(act.size)
        while idx.size:
            p = prv[a[idx]]
            ok = p >= 0
            ps = np.where(ok, p, 0)
            px = X[ps]
            py = Y[ps]
            adv = ok & ((by[idx] - py) * (bx[idx] - ax[idx]) > (by[idx] - ay[idx]) * (bx[idx] - px))
            idx = idx[adv]
            loc = loc[adv]
            a[idx] = p[adv]
            ax[idx] = px[adv]
            ay[idx] = py[adv]
            moved[loc] = True
        act = act[moved]
    return (by - ay) / (bx - ax)


def _lookup_py(c, p0, p1, s):
    """(p0, p1) for scores ``s`` with one binary search per score."""
    import numpy as np

    i = np.searchsorted(c, s, side="left")
    tie = c[np.minimum(i, c.shape[0] - 1)] == s
    return p0[i + tie], p1[i]


if HAVE_NUMBA:  # pragma: no cover
    from numba import prange

    @njit(parallel=True, cache=True)
    def _lookup_nb(c, p0, p1, s, a, b):
        k = c.shape[0]
        for t in prange(s.shape[0]):
            v = s[t]
            lo = 0
            hi = k
            while lo < hi:
                mid = (lo + hi) >> 1
                if c[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            b[t] = p1[lo]
            if lo < k and c[lo] == v:
                a[t] = p0[lo + 1]
            else:
                a[t] = p0[lo]


_PARALLEL_MIN = 20_000


def lookup(c, p0, p1, s):
    """Venn-Abers pair for every score in ``s``."""
    import numpy as np

    if HAVE_NUMBA and s.shape[0] >= _PARALLEL_MIN:
        a = np.empty(s.shape[0])
        b = np.empty(s.shape[0])
        _lookup_nb(c, p0, p1, np.ascontiguousarray(s), a, b)
        return a, b
    return _lookup_py(c, p0, p1, s)
