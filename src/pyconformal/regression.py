"""Split conformal regression."""

from __future__ import annotations

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._quantile import _check_alpha, conformal_quantile, scaled_band, symmetric_band


class SplitConformalRegressor(_Conformal):
    """Split (inductive) conformal prediction intervals for regression.

    The nonconformity score is the absolute residual ``|y - y_pred|`` on a
    held-out calibration set. Intervals are ``y_pred +/- q``, where ``q`` is
    the finite-sample conformal quantile of the calibration scores.

    If the calibration and test data are exchangeable, a new interval
    contains the true value with probability at least ``1 - alpha``
    (marginal coverage). Intervals have the same width for every input,
    unless one of the three options below is used. They are given at
    ``calibrate`` and again at ``predict_interval``:

    * ``sigma``: a positive difficulty estimate per sample (normalized
      conformal prediction). Scores become ``|y - y_pred| / sigma`` and
      intervals ``y_pred +/- q * sigma``, wider where the model is less sure.
    * ``groups``: a category per sample (Mondrian conformal prediction). Each
      category gets its own threshold, so coverage holds within every
      category.
    * ``weights``: for covariate shift (Tibshirani et al., 2019). Each
      sample's weight is how much more likely its inputs are under the test
      distribution than under the calibration distribution. Coverage then
      holds on the shifted test distribution, if the weights are right.

    Works from predictions::

        cp = SplitConformalRegressor(alpha=0.1)
        cp.calibrate(y_pred=pred_cal, y=y_cal)
        cp.predict_interval(y_pred=pred_test)

    or from a model::

        cp = SplitConformalRegressor(model, alpha=0.1)
        cp.fit(X_train, y_train)          # skip if already trained
        cp.calibrate(X_cal, y_cal)
        cp.predict_interval(X_test)

    Parameters
    ----------
    estimator : object or None
        A regressor with ``predict`` (and ``fit``, if it is to be trained
        here). None when working from predictions.
    alpha : float, default=0.1
        Default miscoverage level; 0.1 gives 90% intervals.
    """

    def __init__(self, estimator=None, alpha: float = 0.1):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.calibration_scores_ = None

    def calibrate(self, X=None, y=None, *, y_pred=None, sigma=None, groups=None, weights=None):
        """Compute scores on data not used for training."""
        if y is None:
            raise TypeError("calibrate() needs the outcomes y.")
        y = np.asarray(y, dtype=float).ravel()
        y_hat = self._output(X, y_pred).ravel()
        if y_hat.shape != y.shape:
            raise ValueError("Predictions and y have different lengths.")
        n = y.shape[0]
        scores = np.abs(y - y_hat)
        self._normalized = sigma is not None
        if self._normalized:
            scores = scores / self._positive(sigma, n, "sigma")
        if weights is not None and groups is not None:
            raise ValueError("weights and groups cannot be combined.")
        self._group_scores = None
        if groups is not None:
            g = np.asarray(groups).ravel()
            if g.shape[0] != n:
                raise ValueError("groups has the wrong length.")
            keys, codes = np.unique(g, return_inverse=True)
            order = np.argsort(codes, kind="stable")
            bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=keys.shape[0]))])
            self._group_keys = keys
            self._group_scores = {
                k: scores[order[bounds[j] : bounds[j + 1]]] for j, k in enumerate(keys.tolist())
            }
        self._cum_weights = None
        if weights is not None:
            w = self._positive(weights, n, "weights", allow_zero=True)
            order = np.argsort(scores, kind="stable")
            self._sorted_scores = scores[order]
            self._cum_weights = np.cumsum(w[order])
        self.calibration_scores_ = scores
        self._cache = {}
        return self

    @staticmethod
    def _positive(v, n, name, allow_zero=False):
        v = np.asarray(v, dtype=float).ravel()
        if v.shape[0] != n:
            raise ValueError(f"{name} has the wrong length.")
        lo, hi = (v.min(), v.max()) if v.size else (1.0, 1.0)
        if not (lo >= 0.0) or not np.isfinite(hi) or (not allow_zero and lo == 0.0):
            raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}.")
        return v

    def predict(self, X):
        """Point predictions from the underlying regressor."""
        return self._output(X, None).ravel()

    def threshold(self, alpha: float | None = None) -> float:
        """Half-width of the interval at level ``alpha`` (in units of
        ``sigma`` for normalized intervals). Not defined with groups or
        weights, where it differs between samples."""
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() first.")
        if self._group_scores is not None or self._cum_weights is not None:
            raise RuntimeError("With groups or weights the threshold differs between samples.")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        q = self._cache.get(alpha)
        if q is None:
            q = self._cache[alpha] = conformal_quantile(self.calibration_scores_, alpha)
        return q

    def _thresholds(self, alpha, n, groups, weights):
        """Threshold per sample for the Mondrian and weighted variants."""
        if (groups is None) != (self._group_scores is None):
            raise ValueError("Pass groups at prediction if and only if they were given at calibration.")
        if (weights is None) != (self._cum_weights is None):
            raise ValueError("Pass weights at prediction if and only if they were given at calibration.")
        if groups is not None:
            g = np.asarray(groups).ravel()
            if g.shape[0] != n:
                raise ValueError("groups has the wrong length.")
            keys = self._group_keys
            try:  # one search in the short list of calibration groups
                codes = np.searchsorted(keys, g)
                seen = keys[np.minimum(codes, keys.shape[0] - 1)] == g
            except TypeError:
                seen = np.zeros(n, dtype=bool)
            if not np.all(seen):
                raise ValueError(f"Group {g[~np.asarray(seen)][0]!r} was not seen in calibration.")
            table = self._cache.get(("groups", alpha))
            if table is None:
                table = np.array([conformal_quantile(self._group_scores[k], alpha) for k in keys.tolist()])
                self._cache[("groups", alpha)] = table
            return table[codes]
        # Weighted: the test point carries weight w(x) at +infinity. The
        # threshold is the smallest score whose cumulative calibration weight
        # reaches (1 - alpha) of the total including the test point.
        w = self._positive(weights, n, "weights", allow_zero=True)
        cum = self._cum_weights
        need = (1.0 - alpha) * (cum[-1] + w)
        pos = np.searchsorted(cum, need * (1.0 - 1e-12), side="left")
        q = np.full(n, np.inf)
        ok = pos < cum.shape[0]
        q[ok] = self._sorted_scores[pos[ok]]
        return q

    def predict_interval(self, X=None, alpha=None, *, y_pred=None, sigma=None, groups=None, weights=None):
        """Prediction intervals, shape (n_samples, 2): lower and upper bound.

        ``alpha`` overrides the default level without recalibrating. A list
        of levels gives shape (n_samples, 2, n_levels). Bounds are infinite
        when the calibration set (or the sample's group) is too small for
        ``alpha``.
        """
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() first.")
        alphas, single = self._alphas(alpha)
        y_hat = self._output(X, y_pred).ravel()
        n = y_hat.shape[0]
        if (sigma is None) == self._normalized:
            raise ValueError("Pass sigma at prediction if and only if it was given at calibration.")
        plain = self._group_scores is None and self._cum_weights is None and groups is None and weights is None
        if plain and not self._normalized:
            qs = [self.threshold(a) for a in alphas]
            if single:
                return symmetric_band(y_hat, qs[0])
            return np.stack([symmetric_band(y_hat, q) for q in qs], axis=2)
        scale = self._positive(sigma, n, "sigma") if self._normalized else 1.0
        bands = [
            scaled_band(y_hat, self.threshold(a) if plain else self._thresholds(a, n, groups, weights), scale)
            for a in alphas
        ]
        return bands[0] if single else np.stack(bands, axis=2)
