"""Conformal prediction when the outcome is a whole probability distribution.

In distribution-to-distribution regression each case has an input
distribution and an output distribution, both seen only through samples (the
ages at death in a country this year and next year, say). Following
"Conformal prediction for distribution-to-distribution regression"
(arXiv:2611.06320):

1. each distribution is summarised by its first cosine (Fourier)
   coefficients, :func:`cosine_coefficients`;
2. any regressor predicts the output coefficients from the input ones;
3. split conformal prediction with the Euclidean distance between predicted
   and observed coefficients as the score gives a ball around the prediction
   that contains the observed coefficients with probability at least
   ``1 - alpha``.

:class:`ConformalDistributionRegressor` does step 3 for any vector-valued
prediction, so it also serves ordinary multi-output regression.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._quantile import _check_alpha, conformal_quantile

__all__ = ["ConformalDistributionRegressor", "cosine_coefficients", "cosine_cdf", "cosine_density"]

_SQRT2 = math.sqrt(2.0)


def _coef_py(u, out):
    """Mean of sqrt(2) cos(pi k u) for k = 1 .. K, per row. One cosine per
    point; higher terms follow from cos(k t) = 2 cos(t) cos((k-1) t) -
    cos((k-2) t). Returns 0, or 1 (not finite) or 2 (outside [0, 1])."""
    lo, hi = u.min(), u.max()
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 1
    if lo < -1e-12 or hi > 1.0 + 1e-12:
        return 2
    K = out.shape[1]
    step = max(1, 262144 // max(u.shape[1], 1))  # keep the work arrays in cache
    for s in range(0, u.shape[0], step):
        c1 = np.cos(np.pi * u[s : s + step])
        two = 2.0 * c1
        prev, cur = np.ones_like(c1), c1
        for k in range(K):
            out[s : s + step, k] = _SQRT2 * cur.mean(axis=1)
            if k + 1 < K:
                prev, cur = cur, two * cur - prev
    return 0


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=False)
    def _coef_nb(u, out):
        d, m = u.shape
        K = out.shape[1]
        flag = np.zeros(d, dtype=np.int64)
        for i in prange(d):
            acc = np.zeros(K)
            f = 0
            for j in range(m):
                x = u[i, j]
                if not np.isfinite(x):
                    f = 1
                    break
                if x < -1e-12 or x > 1.0 + 1e-12:
                    f = 2
                    break
                c1 = np.cos(np.pi * x)
                two = 2.0 * c1
                prev = 1.0
                cur = c1
                for k in range(K):
                    acc[k] += cur
                    nxt = two * cur - prev
                    prev = cur
                    cur = nxt
            flag[i] = f
            for k in range(K):
                out[i, k] = 1.4142135623730951 * acc[k] / m
        bad = 0
        for i in range(d):
            if flag[i] == 1:
                return 1
            if flag[i] == 2:
                bad = 2
        return bad

    @njit(parallel=True, cache=True)
    def _local_nb(C, scores, dim, val, left, right, start, end, F, K, ranks, out):
        """For each row of F: the scores of its K nearest rows of C (which is
        stored in k-d tree order), and from them the order statistics
        ``ranks`` (1-based).

        The K best so far sit in a max-heap. A branch of the tree is skipped
        when the splitting plane is already farther than the K-th best, and a
        distance is abandoned once its partial sum exceeds it. Exact."""
        d = C.shape[1]
        for i in prange(F.shape[0]):
            hd = np.empty(K)  # heap of squared distances, largest at 0
            hs = np.empty(K)  # the score that goes with each
            size = 0
            worst = np.inf
            stack = np.empty(128, dtype=np.int64)
            gap = np.empty(128)
            stack[0] = 0
            gap[0] = 0.0
            top = 1
            while top > 0:
                top -= 1
                node = stack[top]
                if gap[top] >= worst:
                    continue
                while dim[node] >= 0:  # walk down, nearer side first
                    diff = F[i, dim[node]] - val[node]
                    if diff < 0.0:
                        near, far = left[node], right[node]
                    else:
                        near, far = right[node], left[node]
                    stack[top] = far
                    gap[top] = diff * diff
                    top += 1
                    node = near
                for j in range(start[node], end[node]):
                    acc = 0.0
                    for t in range(d):
                        e = C[j, t] - F[i, t]
                        acc += e * e
                        if acc >= worst:
                            break
                    if acc >= worst:
                        continue
                    if size < K:  # fill: sift up
                        c = size
                        size += 1
                        while c > 0:
                            parent = (c - 1) >> 1
                            if hd[parent] >= acc:
                                break
                            hd[c] = hd[parent]
                            hs[c] = hs[parent]
                            c = parent
                        hd[c] = acc
                        hs[c] = scores[j]
                        if size == K:
                            worst = hd[0]
                    else:  # replace the largest: sift down
                        c = 0
                        while True:
                            child = 2 * c + 1
                            if child >= K:
                                break
                            if child + 1 < K and hd[child + 1] > hd[child]:
                                child += 1
                            if hd[child] <= acc:
                                break
                            hd[c] = hd[child]
                            hs[c] = hs[child]
                            c = child
                        hd[c] = acc
                        hs[c] = scores[j]
                        worst = hd[0]
            hs.sort()
            for r in range(ranks.shape[0]):
                out[i, r] = hs[ranks[r] - 1]

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


def _unit(x, support):
    lo, hi = float(support[0]), float(support[1])
    if not hi > lo:
        raise ValueError("support must be (low, high) with high > low.")
    return (np.asarray(x, dtype=float) - lo) / (hi - lo), hi - lo


def cosine_coefficients(samples, n_terms: int = 10, support=(0.0, 1.0)):
    """Cosine-series coefficients of distributions seen through samples.

    With the sample rescaled from ``support`` to [0, 1], coefficient ``k``
    (``k = 1 .. n_terms``) is the sample mean of ``sqrt(2) * cos(pi * k * x)``.
    The constant term is always 1 and is left out.

    Parameters
    ----------
    samples : array of shape (n_distributions, n_points), a 1-D array for one
        distribution, or a list of 1-D arrays of different lengths
    n_terms : int, default=10
    support : (low, high)
        Interval that contains the data. Values outside it are an error.

    Returns
    -------
    ndarray of shape (n_distributions, n_terms)
    """
    n_terms = int(n_terms)
    if n_terms < 1:
        raise ValueError("n_terms must be at least 1.")

    def one(block):  # (d, m) -> (d, n_terms)
        u, _ = _unit(block, support)
        if u.size == 0:
            raise ValueError("Samples must be non-empty and finite.")
        u = np.ascontiguousarray(u)
        out = np.empty((u.shape[0], n_terms))
        bad = (_coef_nb if _HAVE_NUMBA else _coef_py)(u, out)
        if bad == 1:
            raise ValueError("Samples must be non-empty and finite.")
        if bad == 2:
            raise ValueError("Samples fall outside the support; pass support=(low, high).")
        return out

    if isinstance(samples, np.ndarray) and samples.dtype != object:
        return one(np.atleast_2d(samples))
    try:
        arr = np.asarray(samples, dtype=float)
        if arr.ndim in (1, 2):
            return one(np.atleast_2d(arr))
    except ValueError:
        pass
    return np.vstack([one(np.asarray(s, dtype=float).ravel()[None, :]) for s in samples])


def _cdf_weights(grid, n_terms, support):
    u, _ = _unit(np.atleast_1d(grid), support)
    u = np.clip(u, 0.0, 1.0)
    k = np.pi * np.arange(1, n_terms + 1)
    return u, _SQRT2 * np.sin(u[:, None] * k[None, :]) / k[None, :]  # (g, n_terms)


def cosine_cdf(coefficients, grid, support=(0.0, 1.0)):
    """Distribution function of the truncated cosine series at the points
    ``grid``: shape (n_distributions, n_grid)."""
    a = np.atleast_2d(np.asarray(coefficients, dtype=float))
    u, W = _cdf_weights(grid, a.shape[1], support)
    return u[None, :] + a @ W.T


def cosine_density(coefficients, grid, support=(0.0, 1.0)):
    """Density of the truncated cosine series at ``grid``: shape
    (n_distributions, n_grid). A truncated series can dip below zero."""
    a = np.atleast_2d(np.asarray(coefficients, dtype=float))
    u, width = _unit(np.atleast_1d(grid), support)
    k = np.pi * np.arange(1, a.shape[1] + 1)
    return (1.0 + a @ (_SQRT2 * np.cos(u[:, None] * k[None, :])).T) / width


class ConformalDistributionRegressor(_Conformal):
    """Conformal balls around vector-valued predictions.

    From predictions::

        A_in, A_out = cosine_coefficients(X_samples), cosine_coefficients(Y_samples)
        cd = ConformalDistributionRegressor(alpha=0.1)
        cd.calibrate(y_pred=model.predict(A_in[cal]), y=A_out[cal])
        centre, radius = cd.predict_ball(y_pred=model.predict(A_in[test]))

    or from a model that maps input coefficients to output coefficients (any
    multi-output regressor)::

        cd = ConformalDistributionRegressor(Ridge(), alpha=0.1)
        cd.fit(A_in[train], A_out[train]).calibrate(A_in[cal], A_out[cal])
        cd.predict_ball(A_in[test])

    The score is the Euclidean distance between prediction and outcome. If
    calibration and test cases are exchangeable, the ball contains the
    outcome vector with probability at least ``1 - alpha``.

    Parameters
    ----------
    estimator : object or None
        A regressor whose ``predict`` returns one row of outputs per case.
    alpha : float, default=0.1
    weights : array-like of shape (n_outputs,) or None
        Positive weight of each output in the score, which becomes
        ``sqrt(sum(weights * (y - y_pred) ** 2))``; the set is then an
        ellipsoid. The paper weights cosine term ``k`` by ``k ** (2 *
        sigma)`` (a smoothness assumption), so that errors in the fine detail
        count for more; its code uses no weights, which is the default here.
    k_neighbors : int or None
        None: one radius for all cases, with the guarantee above. A number
        ``K``: the radius of each case comes from the ``K`` calibration cases
        whose features are nearest to it, so it is wider where the model is
        worse. This is the localised variant of the paper; its coverage is
        approximate, with no finite-sample guarantee. When ``K`` is too small
        for ``alpha`` the radius is infinite.

    What the guarantee is about: the outcome's coefficient vector as computed
    from its sample. It says nothing about detail of the distribution beyond
    the retained coefficients.
    """

    def __init__(self, estimator=None, alpha: float = 0.1, k_neighbors=None, weights=None):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("weights must be finite and positive.")
        self.weights = weights
        if k_neighbors is not None:
            if int(k_neighbors) != k_neighbors or k_neighbors < 1:
                raise ValueError("k_neighbors must be a positive integer or None.")
            k_neighbors = int(k_neighbors)
        self.k_neighbors = k_neighbors
        self.calibration_scores_ = None

    # ------------------------------------------------------------- inputs

    def _pred(self, X, y_pred):
        P = self._output(X, y_pred)
        if P.ndim == 1:
            P = P[:, None]
        if P.ndim != 2 or not np.all(np.isfinite(P)):
            raise ValueError("Predictions must be a finite array of shape (n_samples, n_outputs).")
        return P

    def _w(self, n_outputs):
        if self.weights is None:
            return None
        w = np.asarray(self.weights, dtype=float).ravel()
        if w.shape[0] != n_outputs:
            raise ValueError(f"weights has {w.shape[0]} entries for {n_outputs} outputs.")
        return w

    def _dist(self, D):
        w = self._w(D.shape[1])
        return np.sqrt(np.einsum("ij,ij->i", D, D) if w is None else np.einsum("ij,j,ij->i", D, w, D))

    def _feats(self, X, features, n):
        """Features for the localised variant: ``features=``, else ``X``."""
        if self.k_neighbors is None:
            return None
        F = features if features is not None else X
        if F is None:
            raise TypeError("With k_neighbors, pass the cases' features: X, or features= alongside y_pred=.")
        F = np.asarray(F, dtype=float)
        F = F[:, None] if F.ndim == 1 else F
        if F.ndim != 2 or F.shape[0] != n or not np.all(np.isfinite(F)):
            raise ValueError("features must be a finite array with one row per case.")
        return F

    def calibrate(self, X=None, y=None, *, y_pred=None, features=None):
        """Compute scores on cases not used for training. ``y`` holds the
        observed outcome vectors, one row per case."""
        if y is None:
            raise TypeError("calibrate() needs the outcomes y.")
        P = self._pred(X, y_pred)
        Y = np.asarray(y, dtype=float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        if Y.shape != P.shape:
            raise ValueError("Predictions and y must have the same shape.")
        if not np.all(np.isfinite(Y)):
            raise ValueError("y must be finite.")
        F = self._feats(X, features, P.shape[0])
        if F is not None and F.shape[0] < self.k_neighbors:
            raise ValueError("k_neighbors is larger than the calibration set.")
        self._features = F
        self.n_outputs_ = P.shape[1]
        self.calibration_scores_ = self._dist(Y - P)
        return self

    # ------------------------------------------------------------- radius

    def _check(self):
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() first.")

    def _use_tree(self, n_queries):
        """Tree search pays when few of the calibration cases are neighbours
        and the features vary mostly in a few directions (as cosine
        coefficients of smooth distributions do). Otherwise one matrix
        product per block of cases is faster. Both give the same radii."""
        C, K = self._features, self.k_neighbors
        if C.shape[0] < 20 * K or n_queries * C.shape[0] < 2_000_000:
            return False
        v = C.var(axis=0)
        return float(v.sum()) ** 2 <= 6.0 * float((v * v).sum())  # about six effective directions or fewer

    def _kd_tree(self, leaf: int = 48):
        """k-d tree over the calibration features, built once per
        calibration: rows in tree order, their scores, and the nodes
        (splitting column or -1 for a leaf, splitting value, children, and
        the row range of each leaf)."""
        tree = self.__dict__.get("_tree")
        if tree is not None and tree[0] is self._features:
            return tree[1]
        C = self._features
        n = C.shape[0]
        perm = np.arange(n)
        dim, val, left, right, start, end = [], [], [], [], [], []

        def new(lo, hi):
            dim.append(-1); val.append(0.0); left.append(-1); right.append(-1); start.append(lo); end.append(hi)
            return len(dim) - 1

        todo = [new(0, n)]
        while todo:
            node = todo.pop()
            lo, hi = start[node], end[node]
            if hi - lo <= leaf:
                continue
            block = C[perm[lo:hi]]
            spread = block.max(axis=0) - block.min(axis=0)
            k = int(np.argmax(spread))
            if not spread[k] > 0.0:
                continue  # all rows equal: keep as a leaf
            mid = (hi - lo) // 2
            part = np.argpartition(block[:, k], mid)
            perm[lo:hi] = perm[lo:hi][part]
            dim[node], val[node] = k, float(block[part[mid], k])
            left[node], right[node] = new(lo, lo + mid), new(lo + mid, hi)
            todo += [left[node], right[node]]
        arrays = (np.ascontiguousarray(C[perm]), np.ascontiguousarray(self.calibration_scores_[perm]),
                  np.array(dim, dtype=np.int64), np.array(val), np.array(left, dtype=np.int64),
                  np.array(right, dtype=np.int64), np.array(start, dtype=np.int64), np.array(end, dtype=np.int64))
        self._tree = (C, arrays)
        return arrays

    def _neighbours(self, F):
        """Scores of the K nearest calibration cases, shape (n, K)."""
        C, K = self._features, self.k_neighbors
        if F.shape[1] != C.shape[1]:
            raise ValueError("features has a different number of columns than in calibration.")
        out = np.empty((F.shape[0], K))
        c2 = np.einsum("ij,ij->i", C, C)
        step = max(1, int(4e6 // max(C.shape[0], 1)))
        for s in range(0, F.shape[0], step):
            B = F[s : s + step]
            d = c2[None, :] - 2.0 * (B @ C.T)  # squared distance up to a per-row constant
            idx = np.argpartition(d, K - 1, axis=1)[:, :K] if K < C.shape[0] else np.broadcast_to(np.arange(K), (B.shape[0], K))
            out[s : s + step] = self.calibration_scores_[idx]
        return out

    @staticmethod
    def _ranks(n, alphas):
        return [math.ceil((n + 1) * (1 - Fraction(repr(a)))) for a in alphas]

    def radius(self, X=None, alpha=None, *, features=None):
        """Radius of the ball: one number, or one per case with
        ``k_neighbors``. Infinite when there are too few calibration cases
        for ``alpha``. A list of levels gives one radius per level (last
        axis); the nearest neighbours are then found once for all levels."""
        self._check()
        alphas, single = self._alphas(alpha)
        if self.k_neighbors is None:
            srt = self.__dict__.get("_sorted")
            if srt is None or srt[0] is not self.calibration_scores_:
                srt = self._sorted = (self.calibration_scores_, np.sort(self.calibration_scores_))
            n = srt[1].shape[0]
            out = np.array([np.inf if k > n else srt[1][k - 1] for k in self._ranks(n, alphas)])
            return float(out[0]) if single else out
        F = features if features is not None else X
        if F is None:
            raise TypeError("With k_neighbors, pass the cases' features.")
        F = np.asarray(F, dtype=float)
        F = F[:, None] if F.ndim == 1 else F
        K = self.k_neighbors
        ranks = self._ranks(K, alphas)
        out = np.full((F.shape[0], len(ranks)), np.inf)
        finite = sorted({k - 1 for k in ranks if k <= K})
        if finite and _HAVE_NUMBA and self._use_tree(F.shape[0]):
            if F.shape[1] != self._features.shape[1]:
                raise ValueError("features has a different number of columns than in calibration.")
            cols = [j for j, k in enumerate(ranks) if k <= K]
            part = np.empty((F.shape[0], len(cols)))
            tree = self._kd_tree()
            _local_nb(*tree, np.ascontiguousarray(F), K, np.array([ranks[j] for j in cols], dtype=np.int64), part)
            out[:, cols] = part
        elif finite:
            S = self._neighbours(F)
            S.partition(finite, axis=1)
            for j, k in enumerate(ranks):
                if k <= K:
                    out[:, j] = S[:, k - 1]
        return out[:, 0] if single else out

    def predict(self, X=None, *, y_pred=None):
        """The predicted vectors (centres of the balls)."""
        return self._pred(X, y_pred)

    def predict_ball(self, X=None, alpha=None, *, y_pred=None, features=None):
        """(centres, radii): centres of shape (n_samples, n_outputs) and one
        radius per case."""
        self._check()
        P = self._pred(X, y_pred)
        if P.shape[1] != self.n_outputs_:
            raise ValueError(f"Expected {self.n_outputs_} outputs per case, as in calibration.")
        if self.k_neighbors is None:
            r = np.full(P.shape[0], self.radius(alpha=alpha))
        else:
            r = self.radius(alpha=alpha, features=self._feats(X, features, P.shape[0]))
        return P, r

    def contains(self, y, X=None, alpha=None, *, y_pred=None, features=None):
        """Whether each outcome vector lies in its ball."""
        P, r = self.predict_ball(X, alpha, y_pred=y_pred, features=features)
        Y = np.asarray(y, dtype=float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        if Y.shape != P.shape:
            raise ValueError("Predictions and y must have the same shape.")
        return self._dist(Y - P) <= r

    def log_volume(self, X=None, alpha=None, *, features=None):
        """Natural logarithm of the volume of the set (Theorem 4.3 of the
        paper): ``n/2 log(pi) + n log(radius) - lgamma(n/2 + 1) - 1/2
        sum(log(weights))`` for ``n`` outputs. One number, or one per case
        with ``k_neighbors``. Compare sets of the same ``n`` only."""
        r = self.radius(X, alpha, features=features)
        n = self.n_outputs_
        w = self._w(n)
        with np.errstate(divide="ignore"):
            out = 0.5 * n * math.log(math.pi) + n * np.log(r) - math.lgamma(0.5 * n + 1.0)
        return out - (0.0 if w is None else 0.5 * float(np.sum(np.log(w))))

    def predict_cdf_band(self, grid, X=None, alpha=None, *, y_pred=None, features=None, support=(0.0, 1.0)):
        """A band for the distribution function, for outcomes that are
        cosine coefficients (see :func:`cosine_coefficients`).

        Returns ``(lower, upper)``, each of shape (n_samples, n_grid). Every
        coefficient vector in the ball has its (truncated-series)
        distribution function between the two curves at every grid point at
        once, and each bound is attained by some vector in the ball. So when
        the ball contains the outcome, the band contains the outcome's
        truncated-series distribution function everywhere: coverage of the
        band is at least that of the ball. Read quantile bounds off the band
        by inverting it. The curves are cut to [0, 1]; a truncated series
        can stray slightly outside that range, so compare it after the same
        cut.
        """
        P, r = self.predict_ball(X, alpha, y_pred=y_pred, features=features)
        u, W = _cdf_weights(grid, P.shape[1], support)
        mid = u[None, :] + P @ W.T
        w = self._w(P.shape[1])
        reach = np.linalg.norm(W if w is None else W / np.sqrt(w)[None, :], axis=1)
        with np.errstate(invalid="ignore"):
            half = np.where(np.isinf(r)[:, None], np.inf, r[:, None] * reach[None, :])
        return np.clip(mid - half, 0.0, 1.0), np.clip(mid + half, 0.0, 1.0)
