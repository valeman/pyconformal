"""Jackknife+ and CV+ prediction intervals.

Split conformal prediction sets part of the data aside for calibration. The
jackknife+ and CV+ (Barber, Candes, Ramdas and Tibshirani, 2021) use every
point for both training and calibration: the data are cut into K folds, a
model is trained without each fold, and each point is scored by the model
that did not see it. CV+ with K = n is the jackknife+.

Guarantees under exchangeability, with no assumption on the model:

* ``method="plus"``: coverage at least ``1 - 2 * alpha`` (in practice close
  to ``1 - alpha``).
* ``method="minmax"``: coverage at least ``1 - alpha``, with wider intervals.

Reference
---------
Barber, Candes, Ramdas and Tibshirani (2021). Predictive inference with the
jackknife+. Annals of Statistics 49(1), 486-507.
"""

from __future__ import annotations

import copy
import math
from fractions import Fraction

import numpy as np

from pyconformal._base import _Conformal, _rows
from pyconformal._quantile import _check_alpha

__all__ = ["CrossConformalRegressor"]

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        return lambda f: f

    prange = range


@njit(cache=True)
def _count(p, A, starts, v, strict):
    """How many values p[k] + A[j] (j in fold k) are <= v (or < v)."""
    total = 0
    for k in range(p.shape[0]):
        lo = starts[k]
        hi = starts[k + 1]
        while lo < hi:
            mid = (lo + hi) >> 1
            x = p[k] + A[mid]
            if x < v or (not strict and x == v):
                lo = mid + 1
            else:
                hi = mid
        total += lo - starts[k]
    return total


@njit(cache=True)
def _kth(p, A, starts, j, lo_idx, hi_idx):
    """j-th smallest (1-based) of the values p[k] + A[i], where fold k owns
    A[starts[k]:starts[k+1]], sorted ascending. No list is ever merged: a
    pivot is taken from the widest remaining stretch and counted by binary
    search in every fold."""
    K = p.shape[0]
    for k in range(K):
        lo_idx[k] = starts[k]
        hi_idx[k] = starts[k + 1]
    while True:
        best = -1
        width = 0
        for k in range(K):
            if hi_idx[k] - lo_idx[k] > width:
                width = hi_idx[k] - lo_idx[k]
                best = k
        v = p[best] + A[(lo_idx[best] + hi_idx[best]) >> 1]
        if _count(p, A, starts, v, False) >= j:
            if _count(p, A, starts, v, True) < j:
                return v
            for k in range(K):  # keep values strictly below the pivot
                lo, hi = lo_idx[k], hi_idx[k]
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if p[k] + A[mid] < v:
                        lo = mid + 1
                    else:
                        hi = mid
                hi_idx[k] = lo
        else:
            for k in range(K):  # keep values strictly above the pivot
                lo, hi = lo_idx[k], hi_idx[k]
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if p[k] + A[mid] <= v:
                        lo = mid + 1
                    else:
                        hi = mid
                lo_idx[k] = lo


@njit(parallel=True, cache=True)
def _plus_nb(P, up, down, starts, j_lo, j_hi, out):
    n, K = P.shape
    for i in prange(n):
        a = np.empty(K, dtype=np.int64)
        b = np.empty(K, dtype=np.int64)
        out[i, 0] = -np.inf if j_lo < 1 else _kth(P[i], down, starts, j_lo, a, b)
        out[i, 1] = np.inf if j_hi > up.shape[0] else _kth(P[i], up, starts, j_hi, a, b)


class CrossConformalRegressor(_Conformal):
    """Jackknife+ and CV+ prediction intervals.

    From a model::

        cp = CrossConformalRegressor(model, alpha=0.1, cv=10)
        cp.fit(X, y)                 # trains one model per fold and calibrates
        cp.predict_interval(X_test)

    From predictions, when the fold models were trained elsewhere::

        cp = CrossConformalRegressor(alpha=0.1)
        cp.calibrate(y_pred=oof, y=y, folds=fold_of_each_row)
        cp.predict_interval(y_pred=fold_predictions)   # (n_test, K), column k from model k

    Parameters
    ----------
    estimator : object or None
        A regressor with ``fit`` and ``predict``; it is copied once per fold.
    alpha : float, default=0.1
    cv : int or "loo", default=10
        Number of folds. ``"loo"`` leaves one point out at a time (the
        jackknife+), which trains one model per training point.
    method : {"plus", "minmax"}, default="plus"
    shuffle : bool, default=True
        Assign rows to folds at random. Turn off only if the rows are
        already in random order.
    random_state : int or None
    """

    def __init__(self, estimator=None, alpha: float = 0.1, cv=10, method: str = "plus", shuffle: bool = True, random_state=None):
        if method not in ("plus", "minmax"):
            raise ValueError("method must be 'plus' or 'minmax'.")
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.cv = cv
        self.method = method
        self.shuffle = shuffle
        self.random_state = random_state
        self.calibration_scores_ = None
        self.estimators_ = None

    def fit(self, X, y, **fit_params):
        """Train one copy of the model without each fold, then calibrate on
        the out-of-fold predictions. No separate ``calibrate`` call is
        needed."""
        if self.estimator is None:
            raise TypeError("fit() needs an estimator. With predictions, use calibrate().")
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]
        K = n if self.cv == "loo" else int(self.cv)
        if not 2 <= K <= n:
            raise ValueError("cv must be between 2 and the number of rows, or 'loo'.")
        order = np.random.default_rng(self.random_state).permutation(n) if self.shuffle else np.arange(n)
        folds = np.empty(n, dtype=np.int64)
        folds[order] = np.arange(n) % K
        oof = np.empty(n)
        models = []
        for k in range(K):
            test = folds == k
            m = copy.deepcopy(self.estimator).fit(X[~test], y[~test], **_rows(fit_params, ~test, len(y)))
            oof[test] = np.asarray(m.predict(X[test]), dtype=float).ravel()
            models.append(m)
        self.estimators_ = models
        return self.calibrate(y_pred=oof, y=y, folds=folds)

    def calibrate(self, X=None, y=None, *, y_pred=None, folds=None):
        """Calibrate from out-of-fold predictions: ``y_pred[i]`` must come
        from the model trained without row ``i``, and ``folds[i]`` says which
        model that is (0 .. K - 1). With an estimator, use ``fit``."""
        if X is not None:
            raise TypeError("Cross-conformal calibration needs out-of-fold predictions; with a model, call fit(X, y).")
        if y is None or y_pred is None or folds is None:
            raise TypeError("calibrate() needs y_pred=, y= and folds=.")
        y = np.asarray(y, dtype=float).ravel()
        p = np.asarray(y_pred, dtype=float).ravel()
        f = np.asarray(folds).ravel()
        if not (y.shape == p.shape == f.shape):
            raise ValueError("y_pred, y and folds have different lengths.")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
            raise ValueError("y_pred and y must be finite.")
        keys, codes = np.unique(f, return_inverse=True)
        K = keys.shape[0]
        if K < 2 or not np.array_equal(keys, np.arange(K)):
            raise ValueError("folds must use every label 0 .. K - 1, with K at least 2.")
        r = np.abs(y - p)
        order = np.lexsort((r, codes))  # by fold, then by residual
        self._starts = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=K))]).astype(np.int64)
        self._up = np.ascontiguousarray(r[order])  # residuals ascending within each fold
        down = np.empty_like(self._up)
        for k in range(K):  # minus the residuals, ascending within each fold
            a, b = self._starts[k], self._starts[k + 1]
            down[a:b] = -self._up[a:b][::-1]
        self._down = down
        self.n_folds_ = K
        self.calibration_scores_ = r
        return self

    def _fold_predictions(self, X, y_pred):
        if y_pred is not None:
            if X is not None:
                raise TypeError("Pass X or y_pred=, not both.")
            P = np.asarray(y_pred, dtype=float)
        else:
            if X is None or self.estimators_ is None:
                raise TypeError("Pass X (after fit) or y_pred= with one column per fold model.")
            P = np.column_stack([np.asarray(m.predict(X), dtype=float).ravel() for m in self.estimators_])
        if P.ndim != 2 or P.shape[1] != self.n_folds_:
            raise ValueError(f"Expected predictions of shape (n_samples, {self.n_folds_}): one column per fold model.")
        if not np.all(np.isfinite(P)):
            raise ValueError("Predictions must be finite.")
        return np.ascontiguousarray(P)

    def predict(self, X=None, *, y_pred=None):
        """Point prediction: the average of the fold models."""
        if self.calibration_scores_ is None:
            raise RuntimeError("Call fit() or calibrate() first.")
        return self._fold_predictions(X, y_pred).mean(axis=1)

    def _one(self, P, alpha):
        n = self._up.shape[0]
        a = Fraction(repr(float(alpha)))
        j_hi = math.ceil((1 - a) * (n + 1))  # the upper end is this smallest value of prediction + residual
        if self.method == "minmax":
            if j_hi > n:
                return np.column_stack([np.full(P.shape[0], -np.inf), np.full(P.shape[0], np.inf)])
            q = np.partition(self.calibration_scores_, j_hi - 1)[j_hi - 1]
            return np.column_stack([P.min(axis=1) - q, P.max(axis=1) + q])
        j_lo = math.floor(a * (n + 1))  # the lower end is this smallest value of prediction - residual
        out = np.empty((P.shape[0], 2))
        if HAVE_NUMBA and self.n_folds_ * 8 <= n:
            _plus_nb(P, self._up, self._down, self._starts, j_lo, j_hi, out)
            return out
        # many folds (or no numba): one value per calibration point, selected directly
        sizes = np.diff(self._starts)
        col = np.repeat(np.arange(self.n_folds_), sizes)
        for s in range(0, P.shape[0], 256):
            Pb = P[s : s + 256][:, col]
            out[s : s + 256, 0] = -np.inf if j_lo < 1 else np.partition(Pb + self._down, j_lo - 1, axis=1)[:, j_lo - 1]
            out[s : s + 256, 1] = np.inf if j_hi > n else np.partition(Pb + self._up, j_hi - 1, axis=1)[:, j_hi - 1]
        return out

    def predict_interval(self, X=None, alpha=None, *, y_pred=None):
        """Prediction intervals, shape (n_samples, 2), or
        (n_samples, 2, n_levels) for a list of levels."""
        if self.calibration_scores_ is None:
            raise RuntimeError("Call fit() or calibrate() first.")
        alphas, single = self._alphas(alpha)
        P = self._fold_predictions(X, y_pred)
        if single:
            return self._one(P, alphas[0])
        return np.stack([self._one(P, a) for a in alphas], axis=2)
