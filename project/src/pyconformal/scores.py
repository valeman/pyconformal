"""Nonconformity scores for classification.

A score maps class probabilities to one number per class; lower means the
class conforms better. Every score here computes the scores of all classes
with one routine, and the score of the true label is read from that same
array. Calibration and prediction can therefore never disagree.

With ``pi_(1) >= pi_(2) >= ...`` the sorted probabilities of a sample, ``r``
the rank of a class (1 = most likely) and ``u`` a uniform random number
drawn once per sample:

* LAC:   ``1 - p``
* APS:   ``pi_(1) + ... + pi_(r-1) + u * pi_(r)``
* RAPS:  APS ``+ penalty * max(0, r - k_reg)``
* SAPS:  ``u * pi_(1)`` if ``r = 1``, else ``pi_(1) + (r - 2 + u) * weight``

With ``randomized=False``, ``u`` is fixed at 1. Coverage is at least
``1 - alpha`` either way; randomisation makes it (almost) exactly
``1 - alpha`` and the sets smaller.

References
----------
Sadinle, Lei and Wasserman (2019), LAC. Romano, Sesia and Candes (2020), APS.
Angelopoulos, Bates, Jordan and Malik (2021), RAPS. Huang, Xi, Zhang, Yao,
Qiu and Wei (2024), SAPS.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LAC", "APS", "RAPS", "SAPS", "get_score"]


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    @njit(cache=True)
    def _row_score(kind, a, b, r, p, cum, top, u):
        # kind 0 = APS, 1 = RAPS, 2 = SAPS; r is the 0-based rank
        if kind == 2:
            if r == 0:
                return u * top
            return top + (r - 1 + u) * a
        s = cum + u * p
        if kind == 1 and r + 1 > b:
            s += a * (r + 1 - b)
        return s

    @njit(cache=True)
    def _sift(hv, hi, s, m):
        # heap with the *worst* kept class at the root; a class is worse when
        # its probability is lower, or equal with a higher class index
        while True:
            c = 2 * s + 1
            if c >= m:
                return
            if c + 1 < m and (hv[c + 1] < hv[c] or (hv[c + 1] == hv[c] and hi[c + 1] > hi[c])):
                c += 1
            if hv[c] < hv[s] or (hv[c] == hv[s] and hi[c] > hi[s]):
                hv[s], hv[c] = hv[c], hv[s]
                hi[s], hi[c] = hi[c], hi[s]
                s = c
            else:
                return

    @njit(cache=True)
    def _top(row, m, hv, hi):
        """Fill ``hv``/``hi`` with the ``m`` most likely classes of ``row``,
        most likely first, equal probabilities in class order. One pass over
        the row and a small heap; the row is never fully sorted."""
        K = row.shape[0]
        for j in range(m):
            hv[j] = row[j]
            hi[j] = j
        for s in range(m // 2 - 1, -1, -1):
            _sift(hv, hi, s, m)
        for j in range(m, K):
            if row[j] > hv[0]:
                hv[0] = row[j]
                hi[0] = j
                _sift(hv, hi, 0, m)
        for end in range(m - 1, 0, -1):
            hv[0], hv[end] = hv[end], hv[0]
            hi[0], hi[end] = hi[end], hi[0]
            _sift(hv, hi, 0, end)

    @njit(cache=True)
    def _set_row(row, u, kind, a, b, q, qmax, out, m):
        """Write the set of one row using its top ``m`` classes. Returns the
        number of classes that had to be ranked, or -1 if ``m`` was too few."""
        K = row.shape[0]
        hv = np.empty(m)
        hi = np.empty(m, dtype=np.int64)
        _top(row, m, hv, hi)
        top = hv[0]
        cum = 0.0
        for r in range(m):
            j = hi[r]
            p = hv[r]
            out[j] = _row_score(kind, a, b, r, p, cum, top, u) <= q[j]
            cum += p
            # the smallest score any later rank can have
            if kind == 2:
                floor = top + r * a
            elif kind == 1 and r + 2 > b:
                floor = cum + a * (r + 2 - b)
            else:
                floor = cum
            if floor > qmax:
                return r + 1
        return K if m == K else -1

    @njit(cache=True)
    def _label_row(row, col, u, kind, a, b, m):
        hv = np.empty(m)
        hi = np.empty(m, dtype=np.int64)
        _top(row, m, hv, hi)
        cum = 0.0
        for r in range(m):
            if hi[r] == col:
                return _row_score(kind, a, b, r, hv[r], cum, hv[0], u), r + 1
            cum += hv[r]
        return 0.0, -1

    @njit(cache=True)
    def _start(need, K):
        # first guess for how many classes to rank, from a few pilot rows
        return min(K, max(8, int(1.25 * need) + 1))

    @njit(parallel=True, cache=True)
    def _sets_nb(P, u, kind, a, b, q, out):
        # Scores grow with rank, so a row is finished as soon as the smallest
        # score any later rank could have is above every threshold. Only the
        # top few classes of each row are ever ranked.
        n, K = P.shape
        qmax = -np.inf
        for j in range(K):
            if q[j] != np.inf and q[j] > qmax:
                qmax = q[j]
        pilot = min(n, 32)
        need = 0
        for i in range(pilot):
            need = max(need, _set_row(P[i], u[i], kind, a, b, q, qmax, out[i], K))
        m0 = _start(need, K)
        for i in prange(pilot, n):
            m = m0
            while _set_row(P[i], u[i], kind, a, b, q, qmax, out[i], m) < 0:
                m = min(K, m * 4)
        for j in range(K):
            if q[j] == np.inf:
                out[:, j] = True

    @njit(parallel=True, cache=True)
    def _label_nb(P, cols, u, kind, a, b, out):
        n, K = P.shape
        pilot = min(n, 32)
        ranks = np.empty(pilot, dtype=np.int64)
        for i in range(pilot):
            out[i], ranks[i] = _label_row(P[i], cols[i], u[i], kind, a, b, K)
        m0 = _start(np.sort(ranks)[(3 * pilot) // 4], K)
        for i in prange(pilot, n):
            m = m0
            while True:
                s, r = _label_row(P[i], cols[i], u[i], kind, a, b, m)
                if r > 0:
                    out[i] = s
                    break
                m = min(K, m * 4)

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


class _Score:
    randomized = False
    _kernel = None  # (kind, a, b) for scores with a compiled kernel

    def label_scores(self, proba, cols, u=None):
        """Score of class ``cols[i]`` for each sample. Uses the same routine
        as :meth:`sets`, so calibration and prediction agree exactly."""
        P = np.ascontiguousarray(proba, dtype=np.float64)
        k = self._kernel_args()
        if k is not None:
            out = np.empty(P.shape[0])
            _label_nb(P, np.asarray(cols, dtype=np.int64), self._u(P.shape[0], u), k[0], k[1], k[2], out)
            return out
        return self.scores(P, u)[np.arange(P.shape[0]), cols]

    def sets(self, proba, q, u=None):
        """Boolean prediction sets: classes whose score is at most ``q``
        (a number, or one threshold per class)."""
        P = np.ascontiguousarray(proba, dtype=np.float64)
        k = self._kernel_args()
        if k is not None:
            qv = np.ascontiguousarray(np.broadcast_to(np.asarray(q, dtype=np.float64), (P.shape[1],)))
            out = np.zeros(P.shape, dtype=np.bool_)
            _sets_nb(P, self._u(P.shape[0], u), k[0], k[1], k[2], qv, out)
            return out
        return self.scores(P, u) <= q

    def _kernel_args(self):
        return self._kernel if HAVE_NUMBA else None

    def scores(self, proba, u=None):
        """Scores of every class, shape (n_samples, n_classes).

        ``u`` holds one number in [0, 1] per sample; it is ignored by scores
        that are not randomised.
        """
        raise NotImplementedError

    def _u(self, n, u):
        if not self.randomized or u is None:
            return np.ones(n)
        u = np.asarray(u, dtype=float).ravel()
        if u.shape[0] != n:
            raise ValueError("u must hold one value per sample.")
        return u

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class LAC(_Score):
    """One minus the probability of the class. Smallest sets on average;
    sets can be empty."""

    def scores(self, proba, u=None):
        return 1.0 - np.asarray(proba, dtype=float)


def _sorted(proba):
    """Probabilities sorted in descending order, and the sorting permutation."""
    P = np.asarray(proba, dtype=float)
    # stable, so equal probabilities rank by class index on every code path
    order = np.argsort(-P, axis=1, kind="stable")
    return np.take_along_axis(P, order, axis=1), order


def _unsort(ordered, order):
    """Put scores computed in sorted order back in class order."""
    out = np.empty_like(ordered)
    np.put_along_axis(out, order, ordered, axis=1)
    return out


class APS(_Score):
    """Adaptive prediction sets: cumulative probability down to the class.
    Sets adapt to how hard each sample is."""

    def __init__(self, randomized: bool = True):
        self.randomized = bool(randomized)

    def _kernel_args(self):
        return (0, 0.0, 0) if HAVE_NUMBA else None

    def _ordered(self, srt, u):
        # mass of the classes ranked above, plus u times the class's own mass;
        # written in the same order as the compiled kernel, so both give the
        # same floating-point numbers
        cum = np.cumsum(srt, axis=1)
        if not self.randomized:
            return cum
        out = np.empty_like(cum)
        out[:, 0] = 0.0
        out[:, 1:] = cum[:, :-1]
        out += u[:, None] * srt
        return out

    def scores(self, proba, u=None):
        srt, order = _sorted(proba)
        return _unsort(self._ordered(srt, self._u(srt.shape[0], u)), order)


class RAPS(APS):
    """Regularised APS: adds ``penalty`` for every rank beyond ``k_reg``,
    which keeps unlikely classes out and makes sets smaller and more stable."""

    def __init__(self, penalty: float = 0.01, k_reg: int = 5, randomized: bool = True):
        if penalty < 0 or k_reg < 0:
            raise ValueError("penalty and k_reg must be >= 0.")
        self.penalty = float(penalty)
        self.k_reg = int(k_reg)
        self.randomized = bool(randomized)

    def _kernel_args(self):
        return (1, self.penalty, self.k_reg) if HAVE_NUMBA else None

    def _ordered(self, srt, u):
        r = np.arange(1, srt.shape[1] + 1)
        return super()._ordered(srt, u) + self.penalty * np.maximum(0, r - self.k_reg)[None, :]


class SAPS(_Score):
    """Sorted APS: uses only the top probability and the rank, so the score
    does not depend on the (often unreliable) tail probabilities."""

    def __init__(self, weight: float = 0.2, randomized: bool = True):
        if weight <= 0:
            raise ValueError("weight must be > 0.")
        self.weight = float(weight)
        self.randomized = bool(randomized)

    def _kernel_args(self):
        return (2, self.weight, 0) if HAVE_NUMBA else None

    def scores(self, proba, u=None):
        srt, order = _sorted(proba)
        u = self._u(srt.shape[0], u)[:, None]
        top = srt[:, :1]
        ordered = top + (np.arange(-1, srt.shape[1] - 1)[None, :] + u) * self.weight
        ordered[:, 0] = (u * top)[:, 0]
        return _unsort(ordered, order)


_NAMES = {"lac": LAC, "aps": APS, "raps": RAPS, "saps": SAPS}


def get_score(score):
    """Turn a name ("lac", "aps", "raps", "saps") or a score object into a
    score object."""
    if isinstance(score, _Score):
        return score
    if isinstance(score, str) and score.lower() in _NAMES:
        return _NAMES[score.lower()]()
    raise ValueError(f"score must be one of {sorted(_NAMES)} or a score object, got {score!r}.")
