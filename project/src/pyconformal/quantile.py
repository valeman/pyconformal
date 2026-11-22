"""Conformal regression on top of quantile predictions: CQR, CTI and CIR.

All three start from estimated conditional quantiles of the outcome and
correct them on a calibration set, so the result has guaranteed coverage
whatever the quality of the quantile model.

* CQR widens (or narrows) one lower and one upper quantile by a constant.
* CTI keeps the interquantile intervals that are short, that is, where the
  estimated density is high. The result can be a union of intervals.
* CIR grows one interval outwards from the shortest interquantile interval,
  always taking the shorter neighbour. The result is a single interval that
  follows skewed distributions.

Each works from quantile predictions (``quantiles=``, one row per sample) or
from a quantile model.

References
----------
Romano, Patterson and Candes (2019). Conformalized quantile regression.
NeurIPS.

Luo and Zhou (2025). Conformal thresholded intervals for efficient
regression. AAAI.

Guo, Luo and Zhou (2026). Fast conformal prediction using conditional
interquantile intervals. AAAI. arXiv:2601.02769.
"""

from __future__ import annotations

import warnings

import math

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._quantile import _check_alpha, conformal_quantile

__all__ = [
    "ConformalizedQuantileRegressor",
    "ConformalThresholdedIntervals",
    "ConformalInterquantileRegressor",
]

_ALMOST_ONE = math.nextafter(1.0, 0.0)


class _QuantileBased(_Conformal):
    """Shared input handling: a matrix of quantile predictions per sample."""

    _given = "quantiles"
    _min_columns = 2
    _exact_columns = None

    def _levels(self):
        raise NotImplementedError

    def fit(self, X, y, **fit_params):
        """Train the quantile model(s) on the proper training set."""
        if isinstance(self.estimator, (tuple, list)):
            for m in self.estimator:
                m.fit(X, y, **fit_params)
            return self
        return super().fit(X, y, **fit_params)

    def _output(self, X, given):
        if given is None and X is not None and self.estimator is not None:
            est = self.estimator
            if isinstance(est, (tuple, list)):
                Q = np.column_stack([np.asarray(m.predict(X), dtype=float).ravel() for m in est])
            else:
                Q = np.asarray(est.predict(X, quantiles=list(self._levels())), dtype=float)
        else:
            Q = super()._output(X, given)
        if Q.ndim != 2 or Q.shape[1] < self._min_columns:
            raise ValueError(
                f"Quantile predictions must have shape (n_samples, k) with k >= {self._min_columns}."
            )
        if self._exact_columns is not None and Q.shape[1] != self._exact_columns:
            raise ValueError(f"Expected {self._exact_columns} quantile columns, got {Q.shape[1]}.")
        if self._check_finite_here and not np.all(np.isfinite(Q)):
            raise ValueError("Quantile predictions must be finite.")
        return Q

    _check_finite_here = True  # subclasses may fold the check into a kernel

    def calibrate_oob(self, y):
        """Calibrate on the forest's own training rows, using out-of-bag
        predictions, so no data has to be set aside. ``y`` is the training
        outcomes, in the order given to ``fit``.

        Needs an estimator with ``predict_oob`` (``QuantileForest``).
        Out-of-bag predictions use about a third of the trees while new
        predictions use all of them, so coverage is close to the target in
        practice but is not guaranteed in finite samples the way a separate
        calibration set is.
        """
        est = self.estimator
        if est is None or isinstance(est, (tuple, list)) or not hasattr(est, "predict_oob"):
            raise TypeError("calibrate_oob() needs an estimator with predict_oob, such as QuantileForest.")
        Q = np.asarray(est.predict_oob(quantiles=list(self._levels())), dtype=float)
        return self.calibrate(quantiles=Q, y=y)

    @staticmethod
    def _outcomes(y, n):
        if y is None:
            raise TypeError("calibrate() needs the outcomes y.")
        y = np.asarray(y, dtype=float).ravel()
        if y.shape[0] != n:
            raise ValueError("Quantile predictions and y have different numbers of rows.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite.")
        return y

    def _check(self):
        if getattr(self, "calibration_scores_", None) is None:
            raise RuntimeError("Call calibrate() first.")


# --------------------------------------------------------------------- CQR


class ConformalizedQuantileRegressor(_QuantileBased):
    """Conformalized quantile regression (CQR).

    Starts from a predicted lower and upper quantile and shifts both by a
    constant found on the calibration set. Interval width follows the
    quantile model, so it adapts to how uncertain each sample is.

    From predictions::

        cqr = ConformalizedQuantileRegressor(alpha=0.1)
        cqr.calibrate(quantiles=q_cal, y=y_cal)      # q_cal: (n, 2) lower, upper
        cqr.predict_interval(quantiles=q_test)

    From a model: pass a pair ``(lower_model, upper_model)``, or one model
    whose ``predict(X, quantiles=[...])`` returns one column per level.

    Parameters
    ----------
    estimator : pair of models, quantile model or None
    alpha : float, default=0.1
        Default miscoverage level.
    symmetric : bool, default=True
        True: one correction for both ends (Romano et al.), from the score
        ``max(lower - y, y - upper)``. False: each end gets its own
        correction at level ``alpha / 2``; coverage still holds and the
        miss rate is balanced between the two sides, at the cost of
        slightly wider intervals.
    quantile_levels : (float, float) or None
        Levels asked from a single quantile model. Default
        ``(alpha / 2, 1 - alpha / 2)``.
    """

    _exact_columns = 2

    def __init__(self, estimator=None, alpha: float = 0.1, symmetric: bool = True, quantile_levels=None):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.symmetric = bool(symmetric)
        self.quantile_levels = quantile_levels
        self.calibration_scores_ = None

    def _levels(self):
        if self.quantile_levels is not None:
            return tuple(self.quantile_levels)
        return (self.alpha / 2.0, 1.0 - self.alpha / 2.0)

    def calibrate(self, X=None, y=None, *, quantiles=None):
        """Compute scores on data not used for training."""
        Q = self._output(X, quantiles)
        y = self._outcomes(y, Q.shape[0])
        if HAVE_NUMBA and y.shape[0] >= _PARALLEL_MIN:
            lo, hi, both = np.empty(y.shape[0]), np.empty(y.shape[0]), np.empty(y.shape[0])
            _cqr_scores_nb(Q[:, 0], Q[:, 1], y, lo, hi, both)
        else:
            lo, hi = Q[:, 0] - y, y - Q[:, 1]
            both = np.maximum(lo, hi)
        self._lo_scores, self._hi_scores = lo, hi
        self.calibration_scores_ = both
        return self

    def corrections(self, alpha: float | None = None):
        """(lower, upper) amounts by which the two quantiles are moved
        outwards. Negative values move them inwards."""
        self._check()
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        if self.symmetric:
            q = conformal_quantile(self.calibration_scores_, alpha)
            return q, q
        return (
            conformal_quantile(self._lo_scores, alpha / 2.0),
            conformal_quantile(self._hi_scores, alpha / 2.0),
        )

    def predict_interval(self, X=None, alpha=None, *, quantiles=None):
        """Prediction intervals, shape (n_samples, 2), or
        (n_samples, 2, n_levels) for a list of levels.

        When a correction is negative and the two ends cross, the interval
        collapses to the crossing midpoint.
        """
        alphas, single = self._alphas(alpha)
        cs = [self.corrections(a) for a in alphas]
        fast = HAVE_NUMBA and single
        self._check_finite_here = not fast
        try:
            Q = self._output(X, quantiles)
        finally:
            self._check_finite_here = True
        if fast and Q.shape[0] >= _PARALLEL_MIN:
            band = np.empty((Q.shape[0], 2))
            if _cqr_band_nb(Q[:, 0], Q[:, 1], cs[0][0], cs[0][1], band):
                raise ValueError("Quantile predictions must be finite.")
            return band
        if fast and not np.all(np.isfinite(Q)):
            raise ValueError("Quantile predictions must be finite.")
        out = np.empty((Q.shape[0], 2, len(cs)))
        for j, (cl, ch) in enumerate(cs):
            lo, hi = Q[:, 0] - cl, Q[:, 1] + ch
            crossed = lo > hi
            if np.any(crossed):
                mid = 0.5 * (lo + hi)
                lo = np.where(crossed, mid, lo)
                hi = np.where(crossed, mid, hi)
            out[:, 0, j], out[:, 1, j] = lo, hi
        return out[:, :, 0] if single else out


# ------------------------------------------------------- interquantile base


class _Interquantile(_QuantileBased):
    """K + 1 quantile predictions per sample define K interquantile intervals
    ``[q_0, q_1], (q_1, q_2], ..., (q_{K-1}, q_K]`` plus two infinite tails.
    Predicted quantiles are sorted within each sample first, so crossing
    quantiles cannot produce negative lengths."""

    def __init__(self, estimator=None, alpha: float = 0.1, quantile_levels=None):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.quantile_levels = quantile_levels
        self.calibration_scores_ = None

    def _levels(self):
        if self.quantile_levels is None:
            return np.linspace(0.0, 1.0, 101)
        if np.isscalar(self.quantile_levels):
            return np.linspace(0.0, 1.0, int(self.quantile_levels) + 1)
        return np.asarray(self.quantile_levels, dtype=float)

    _check_finite_here = False  # done together with the sortedness check

    def _sorted(self, X, quantiles):
        Q = self._output(X, quantiles)
        if HAVE_NUMBA and Q.shape[0] >= 1000:
            Q = np.ascontiguousarray(Q)
            status = np.empty(Q.shape[0], dtype=np.int8)
            _row_status_nb(Q, status)
            if np.any(status == 2):
                raise ValueError("Quantile predictions must be finite.")
            crossed = np.flatnonzero(status == 1)
            if crossed.size:  # sort only the rows whose quantiles cross
                Q = Q.copy()
                Q[crossed] = np.sort(Q[crossed], axis=1)
        else:
            if not np.all(np.isfinite(Q)):
                raise ValueError("Quantile predictions must be finite.")
            if np.any(Q[:, 1:] < Q[:, :-1]):
                Q = np.sort(Q, axis=1)
        if getattr(self, "n_intervals_", None) not in (None, Q.shape[1] - 1):
            raise ValueError("The number of quantile columns differs from calibration.")
        return Q

    @staticmethod
    def _bin(Q, y):
        """Interval holding ``y``: 0 = lower tail, 1..K, K + 1 = upper tail."""
        K = Q.shape[1] - 1
        if HAVE_NUMBA and Q.shape[0] >= 1000:
            b = np.empty(Q.shape[0], dtype=np.int64)
            _bins_nb(Q, y, b)
            return b, K
        b = (Q < y[:, None]).sum(axis=1)  # y in (q_{b-1}, q_b]
        b[(b == 0) & (y >= Q[:, 0])] = 1  # the first interval is closed on the left
        return b, K


# --------------------------------------------------------------------- CTI


class ConformalThresholdedIntervals(_Interquantile):
    """Conformal thresholded intervals (CTI; Luo and Zhou, 2025).

    Each interquantile interval holds about the same probability, so a short
    one means high density. CTI keeps every interval whose length is at most
    a threshold found on the calibration set, which aims for the smallest
    possible prediction set. The set can be a union of separate intervals,
    for example for a bimodal outcome.

    The score of a calibration point is the length of the interval its
    outcome falls in; outcomes beyond the lowest or highest predicted
    quantile fall in a tail of infinite length.

    Parameters
    ----------
    estimator : quantile model or None
        A model whose ``predict(X, quantiles=[...])`` returns one column per
        level. None when working from quantile predictions.
    alpha : float, default=0.1
    quantile_levels : int, array-like or None
        Levels asked from the model: an array, or a number K for K equal
        parts. Default: 100 equal parts (levels 0, 0.01, ..., 1).
    """

    def calibrate(self, X=None, y=None, *, quantiles=None):
        """Compute scores on data not used for training."""
        Q = self._sorted(X, quantiles)
        y = self._outcomes(y, Q.shape[0])
        b, K = self._bin(Q, y)
        length = np.full(y.shape[0], np.inf)
        inside = (b >= 1) & (b <= K)
        rows = np.flatnonzero(inside)
        length[rows] = Q[rows, b[rows]] - Q[rows, b[rows] - 1]
        self.n_intervals_ = K
        self.calibration_scores_ = length
        return self

    def threshold(self, alpha: float | None = None) -> float:
        """Longest interquantile interval that is kept."""
        self._check()
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        return conformal_quantile(self.calibration_scores_, alpha, allow_inf=True)

    def predict_set(self, X=None, alpha=None, *, quantiles=None):
        """Which intervals are in the prediction set: a boolean array of
        shape (n_samples, K + 2). Column 0 is the lower tail, columns 1..K
        the interquantile intervals, column K + 1 the upper tail. A list of
        levels adds a last axis."""
        alphas, single = self._alphas(alpha)
        ts = [self.threshold(a) for a in alphas]
        Q = self._sorted(X, quantiles)
        if HAVE_NUMBA and single and Q.shape[0] >= 1000:
            out = np.empty((Q.shape[0], Q.shape[1] + 1), dtype=np.bool_)
            _cti_sets_nb(Q, ts[0], math.isinf(ts[0]), out)
            return out
        L = np.diff(Q, axis=1)
        out = np.empty((Q.shape[0], L.shape[1] + 2, len(ts)), dtype=bool)
        for j, t in enumerate(ts):
            out[:, 1:-1, j] = L <= t
            out[:, 0, j] = out[:, -1, j] = math.isinf(t)
        return out[:, :, 0] if single else out

    def predict_size(self, X=None, alpha=None, *, quantiles=None):
        """Total length of the prediction set per sample, shape (n_samples,)."""
        t = self.threshold(alpha)
        if math.isinf(t):
            return np.full(self._sorted(X, quantiles).shape[0], np.inf)
        L = np.diff(self._sorted(X, quantiles), axis=1)
        return np.where(L <= t, L, 0.0).sum(axis=1)

    def predict_interval(self, X=None, alpha=None, *, quantiles=None):
        """Smallest single interval containing the prediction set, shape
        (n_samples, 2). It covers at least as often as the set, but can be
        much wider when the set has gaps. Rows are NaN when the set is
        empty."""
        t = self.threshold(alpha)
        Q = self._sorted(X, quantiles)
        n, K = Q.shape[0], Q.shape[1] - 1
        if math.isinf(t):
            return np.column_stack([np.full(n, -np.inf), np.full(n, np.inf)])
        keep = np.diff(Q, axis=1) <= t
        any_ = keep.any(axis=1)
        first = np.argmax(keep, axis=1)
        last = K - 1 - np.argmax(keep[:, ::-1], axis=1)
        rows = np.arange(n)
        out = np.column_stack([Q[rows, first], Q[rows, last + 1]])
        out[~any_] = np.nan
        return out


# --------------------------------------------------------------------- CIR


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    @njit(cache=True)
    def _grow(q, K, lo, hi):
        """One greedy step for one row. Returns (lo, hi, length added)."""
        left = q[lo] - q[lo - 1] if lo > 0 else np.inf
        right = q[hi + 2] - q[hi + 1] if hi < K - 1 else np.inf
        if left <= right:
            return lo - 1, hi, left
        return lo, hi + 1, right

    @njit(cache=True)
    def _start(q, K):
        best = 0
        shortest = q[1] - q[0]
        for j in range(1, K):
            d = q[j + 1] - q[j]
            if d < shortest:
                shortest = d
                best = j
        return best, shortest

    @njit(parallel=True, cache=True)
    def _cir_scores_nb(Q, b, k_out, len_out):
        n, K = Q.shape[0], Q.shape[1] - 1
        for i in prange(n):
            target = b[i] - 1
            if target < 0 or target >= K:
                k_out[i] = K + 1
                len_out[i] = np.inf
                continue
            lo, added = _start(Q[i], K)
            hi = lo
            k = 1
            while target < lo or target > hi:
                lo, hi, added = _grow(Q[i], K, lo, hi)
                k += 1
            k_out[i] = k
            len_out[i] = added

    @njit(parallel=True, cache=True)
    def _cir_windows_nb(Q, f, plus, s, almost_one, out):
        # f steps for every row; with ``plus``, one more where f + e <= s
        n, K = Q.shape[0], Q.shape[1] - 1
        for i in prange(n):
            q = Q[i]
            lo, added = _start(q, K)
            hi = lo
            if f >= 1:
                for _ in range(f - 1):
                    lo, hi, added = _grow(q, K, lo, hi)
                out[i, 0] = q[lo]
                out[i, 1] = q[hi + 1]
            if plus:
                if f >= 1:
                    lo2, hi2, added = _grow(q, K, lo, hi)
                else:
                    lo2, hi2 = lo, hi
                span = q[K] - q[0]
                e = min(added / span, almost_one) if span > 0 else 0.0
                if f + e <= s:
                    out[i, 0] = q[lo2]
                    out[i, 1] = q[hi2 + 1]

    @njit(parallel=True, cache=True)
    def _row_status_nb(Q, status):
        # 0 = finite and sorted, 1 = finite but not sorted, 2 = not finite
        n, m = Q.shape
        for i in prange(n):
            q = Q[i]
            ok = np.isfinite(q[0]) and np.isfinite(q[m - 1])
            for j in range(m - 1):
                if not (q[j + 1] >= q[j]):
                    ok = False
                    break
            if ok:
                status[i] = 0
            else:
                status[i] = 1
                for j in range(m):
                    if not np.isfinite(q[j]):
                        status[i] = 2
                        break

    @njit(parallel=True, cache=True)
    def _bins_nb(Q, y, b):
        # number of quantiles strictly below y, by binary search
        n, m = Q.shape
        for i in prange(n):
            q = Q[i]
            v = y[i]
            lo = 0
            hi = m
            while lo < hi:
                mid = (lo + hi) >> 1
                if q[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0 and v >= q[0]:
                lo = 1
            b[i] = lo

    @njit(parallel=True, cache=True)
    def _cti_sets_nb(Q, t, tails, out):
        n, K = Q.shape[0], Q.shape[1] - 1
        for i in prange(n):
            q = Q[i]
            out[i, 0] = tails
            out[i, K + 1] = tails
            for j in range(K):
                out[i, j + 1] = q[j + 1] - q[j] <= t

    @njit(parallel=True, cache=True)
    def _cqr_scores_nb(q_lo, q_hi, y, lo, hi, both):
        bad = 0
        for i in prange(y.shape[0]):
            a = q_lo[i] - y[i]
            c = y[i] - q_hi[i]
            lo[i] = a
            hi[i] = c
            both[i] = a if a > c else c
            if not (np.isfinite(a) and np.isfinite(c)):
                bad += 1
        return bad

    @njit(parallel=True, cache=True)
    def _cqr_band_nb(q_lo, q_hi, cl, ch, out):
        # the two columns are passed separately, so any memory layout works
        # without a copy
        bad = 0
        for i in prange(q_lo.shape[0]):
            a = q_lo[i] - cl
            c = q_hi[i] + ch
            if not (np.isfinite(q_lo[i]) and np.isfinite(q_hi[i])):
                bad += 1
            if a > c:
                a = c = 0.5 * (a + c)
            out[i, 0] = a
            out[i, 1] = c
        return bad

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

_PARALLEL_MIN = 20_000


def _greedy(L, snapshots=(), upto=None):
    """Grow a window from the shortest interval of each row, always adding
    the shorter neighbour (the left one on a tie).

    Returns ``step`` (n, K): the step, counted from 1, at which each interval
    joins; ``added`` (n, K): the length added at each step; and for every
    step listed in ``snapshots`` the window (first, last) at that step.
    With ``upto``, stops after that many steps (later entries are unset).
    """
    n, K = L.shape
    rows = np.arange(n)
    lo = np.argmin(L, axis=1)
    hi = lo.copy()
    step = np.empty((n, K), dtype=np.int32)
    added = np.empty((n, K))
    step[rows, lo] = 1
    added[:, 0] = L[rows, lo]
    snaps = {1: (lo.copy(), hi.copy())} if 1 in snapshots else {}
    for k in range(2, (K if upto is None else min(K, upto)) + 1):
        left = np.where(lo > 0, L[rows, np.maximum(lo - 1, 0)], np.inf)
        right = np.where(hi < K - 1, L[rows, np.minimum(hi + 1, K - 1)], np.inf)
        go_left = left <= right
        lo = lo - go_left
        hi = hi + ~go_left
        step[rows, np.where(go_left, lo, hi)] = k
        added[:, k - 1] = np.minimum(left, right)
        if k in snapshots:
            snaps[k] = (lo.copy(), hi.copy())
    return step, added, snaps


class ConformalInterquantileRegressor(_Interquantile):
    """Conformal interquantile regression (CIR and CIR+; Guo, Luo and Zhou,
    2026).

    For each sample a window starts at the shortest interquantile interval
    (the estimated mode) and grows one interval at a time, always taking the
    shorter neighbour. The score of a calibration point is the number of
    intervals the window needs to reach its outcome. The prediction interval
    for a new sample is its window after ``k`` steps, where ``k`` is the
    conformal quantile of the calibration scores. The result is one
    interval that leans towards the dense side of a skewed distribution,
    which equal-tailed CQR cannot do.

    With ``plus=True`` (CIR+) the score also carries the relative length of
    the last interval added, so the final step is taken only for samples
    where it is cheap. Intervals get slightly shorter.

    Coverage is at least ``1 - alpha`` under exchangeability. Two details
    are fixed here so that this holds exactly:

    * The prediction interval is built with the same growing rule as the
      score. The paper's formula instead takes the shortest run of ``k``
      consecutive intervals, which is not always the same window;
      ``window="shortest"`` gives that variant. It is there to reproduce
      the authors' code and has **no coverage guarantee**: on a two-mode
      outcome it covered 88% at a 90% target, and on a simple example with
      exchangeable data it can cover 0% (calibration counts steps along one
      sequence of windows, prediction can pick a different window of the
      same length). It warns when used.
    * For CIR+ the added length is divided by the sample's full predicted
      range, and the extra step is taken when the resulting score is at
      most the threshold. The authors' code divides by 100 times the
      largest calibration outcome instead, which depends on the units and
      sign of the outcome.

    ``window="shortest"`` returns the same intervals as the authors' code
    (github.com/orince/CIR) when the calibration size is such that the two
    rank rules agree; their rank is one higher when ``(1 - alpha) * (n + 1)``
    is a whole number.

    Parameters
    ----------
    estimator : quantile model or None
        A model whose ``predict(X, quantiles=[...])`` returns one column per
        level. None when working from quantile predictions.
    alpha : float, default=0.1
    plus : bool, default=False
        Use CIR+.
    window : {"greedy", "shortest"}, default="greedy"
        "greedy" carries the coverage guarantee. "shortest" reproduces the
        authors' code, without it.
    quantile_levels : int, array-like or None
        Levels asked from the model: an array, or a number K for K equal
        parts. Default: 100 equal parts (levels 0, 0.01, ..., 1).
    """

    def __init__(
        self,
        estimator=None,
        alpha: float = 0.1,
        plus: bool = False,
        window: str = "greedy",
        quantile_levels=None,
    ):
        super().__init__(estimator, alpha, quantile_levels)
        if window not in ("greedy", "shortest"):
            raise ValueError("window must be 'greedy' or 'shortest'.")
        if plus and window == "shortest":
            raise ValueError("CIR+ is defined for window='greedy' only.")
        if window == "shortest":
            warnings.warn(
                "window='shortest' reproduces the CIR authors' code and has no coverage guarantee; "
                "it can cover far less than 1 - alpha. Use the default window='greedy' for the guarantee.",
                UserWarning, stacklevel=2)
        self.plus = bool(plus)
        self.window = window

    @staticmethod
    def _fraction(added, Q):
        """Added length as a fraction of the full predicted range, in [0, 1)."""
        span = (Q[:, -1] - Q[:, 0])[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(span > 0, added / span, 0.0)
        return np.minimum(e, _ALMOST_ONE)

    def calibrate(self, X=None, y=None, *, quantiles=None):
        """Compute scores on data not used for training."""
        Q = self._sorted(X, quantiles)
        y = self._outcomes(y, Q.shape[0])
        b, K = self._bin(Q, y)
        inside = (b >= 1) & (b <= K)
        rows = np.flatnonzero(inside)
        if HAVE_NUMBA:
            k = np.empty(y.shape[0], dtype=np.int64)
            last = np.empty(y.shape[0])  # length added at the final step
            _cir_scores_nb(Q, b.astype(np.int64), k, last)
        else:
            step, added, _ = _greedy(np.diff(Q, axis=1))
            k = np.full(y.shape[0], K + 1, dtype=np.int64)  # tails: one step past the last interval
            k[rows] = step[rows, b[rows] - 1]
            last = np.full(y.shape[0], np.inf)
            last[rows] = added[rows, k[rows] - 1]
        if self.plus:
            e = np.full(y.shape[0], _ALMOST_ONE)
            e[rows] = self._fraction(last[rows, None], Q[rows])[:, 0]
            scores = (k - 1) + e
        else:
            scores = k.astype(float)
        self.n_intervals_ = K
        self.calibration_scores_ = scores
        return self

    def threshold(self, alpha: float | None = None) -> float:
        """Conformal quantile of the calibration scores. For CIR this is the
        number of interquantile intervals in every prediction interval."""
        self._check()
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        return conformal_quantile(self.calibration_scores_, alpha, allow_inf=True)

    def _one(self, Q, s):
        n, K = Q.shape[0], Q.shape[1] - 1
        everything = np.column_stack([np.full(n, -np.inf), np.full(n, np.inf)])
        if math.isinf(s) or s >= K + 1:
            return everything
        rows = np.arange(n)
        L = np.diff(Q, axis=1)
        f = int(math.floor(s))
        if not self.plus:
            if self.window == "shortest":
                width = Q[:, f:] - Q[:, : K + 1 - f]
                l = np.argmin(width, axis=1)
                return np.column_stack([Q[rows, l], Q[rows, l + f]])
            if HAVE_NUMBA:
                out = np.empty((n, 2))
                _cir_windows_nb(np.ascontiguousarray(Q), f, False, s, _ALMOST_ONE, out)
                return out
            lo, hi = _greedy(L, (f,), upto=f)[2][f]
            return np.column_stack([Q[rows, lo], Q[rows, hi + 1]])
        # CIR+: f steps for everyone, one more where the next step is cheap
        if HAVE_NUMBA and f < K:
            out = np.full((n, 2), np.nan)
            _cir_windows_nb(np.ascontiguousarray(Q), f, True, s, _ALMOST_ONE, out)
            return out
        _, added, snaps = _greedy(L, (f, f + 1), upto=f + 1)
        out = np.full((n, 2), np.nan)  # f = 0 and no extra step: empty
        if f >= 1:
            lo, hi = snaps[f]
            out[:, 0], out[:, 1] = Q[rows, lo], Q[rows, hi + 1]
        if f < K:
            more = (f + self._fraction(added, Q)[:, f]) <= s
            lo, hi = snaps[f + 1]
            out[more, 0], out[more, 1] = Q[more, lo[more]], Q[more, hi[more] + 1]
        else:  # f == K: the next step is the tails
            more = (f + _ALMOST_ONE) <= s
            if more:
                return everything
        return out

    def predict_interval(self, X=None, alpha=None, *, quantiles=None):
        """Prediction intervals, shape (n_samples, 2), or
        (n_samples, 2, n_levels) for a list of levels. Bounds are infinite
        when the calibration set is too small for ``alpha`` or too many
        calibration outcomes lie beyond the predicted quantiles."""
        alphas, single = self._alphas(alpha)
        ss = [self.threshold(a) for a in alphas]
        Q = self._sorted(X, quantiles)
        if single:
            return self._one(Q, ss[0])
        return np.stack([self._one(Q, s) for s in ss], axis=2)
