"""Split conformal regression."""

from __future__ import annotations

import numpy as np

from pyconformal._quantile import _check_alpha, conformal_quantile


class SplitConformalRegressor:
    """Split (inductive) conformal prediction intervals for regression.

    Wraps any regressor with ``fit(X, y)`` and ``predict(X)`` methods. The
    nonconformity score is the absolute residual ``|y - y_hat|`` on a held-out
    calibration set. Intervals are ``y_hat +/- q``, where ``q`` is the
    finite-sample conformal quantile of the calibration scores.

    If the calibration and test data are exchangeable, a new interval
    contains the true value with probability at least ``1 - alpha``
    (marginal coverage). Intervals have the same width for every input.

    Parameters
    ----------
    estimator : object
        A regressor exposing ``fit`` and ``predict``.
    alpha : float, default=0.1
        Default miscoverage level; 0.1 gives 90% intervals.
    prefit : bool, default=False
        If True, ``estimator`` is already trained and ``fit`` must not be
        called; go straight to ``calibrate``.
    """

    def __init__(self, estimator, alpha: float = 0.1, prefit: bool = False):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.prefit = prefit
        self.calibration_scores_ = None

    def fit(self, X, y):
        """Train the underlying regressor on the proper training set."""
        if self.prefit:
            raise RuntimeError("prefit=True: the estimator is already trained.")
        self.estimator.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        """Compute absolute-residual scores on data not used for training."""
        y_cal = np.asarray(y_cal, dtype=float).ravel()
        y_hat = np.asarray(self.estimator.predict(X_cal), dtype=float).ravel()
        if y_hat.shape != y_cal.shape:
            raise ValueError("X_cal and y_cal have different numbers of rows.")
        self.calibration_scores_ = np.abs(y_cal - y_hat)
        return self

    def predict(self, X):
        """Point predictions from the underlying regressor."""
        return np.asarray(self.estimator.predict(X), dtype=float).ravel()

    def predict_interval(self, X, alpha: float | None = None):
        """Return prediction intervals as an array of shape (n_samples, 2).

        Column 0 is the lower bound, column 1 the upper bound. Bounds are
        infinite when the calibration set is too small for ``alpha``.
        """
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        q = conformal_quantile(self.calibration_scores_, alpha)
        y_hat = self.predict(X)
        return np.column_stack([y_hat - q, y_hat + q])
