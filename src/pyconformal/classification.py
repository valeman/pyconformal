"""Split conformal classification."""

from __future__ import annotations

import numpy as np

from pyconformal._quantile import _check_alpha, conformal_quantile


class SplitConformalClassifier:
    """Split (inductive) conformal prediction sets for classification.

    Wraps any classifier with ``fit``, ``predict_proba`` and ``classes_``.
    The nonconformity score is ``1 - p_hat(y | x)``, one minus the predicted
    probability of the true class (the LAC score of Sadinle, Lei and
    Wasserman, 2019). A class enters the prediction set when its score is at
    most the finite-sample conformal quantile of the calibration scores.

    If the calibration and test data are exchangeable, the prediction set
    contains the true label with probability at least ``1 - alpha``
    (marginal coverage). Sets can be empty or contain several classes.

    Parameters
    ----------
    estimator : object
        A classifier exposing ``fit``, ``predict_proba`` and ``classes_``.
    alpha : float, default=0.1
        Default miscoverage level; 0.1 gives 90% prediction sets.
    prefit : bool, default=False
        If True, ``estimator`` is already trained and ``fit`` must not be
        called; go straight to ``calibrate``.
    """

    def __init__(self, estimator, alpha: float = 0.1, prefit: bool = False):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.prefit = prefit
        self.calibration_scores_ = None

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit(self, X, y):
        """Train the underlying classifier on the proper training set."""
        if self.prefit:
            raise RuntimeError("prefit=True: the estimator is already trained.")
        self.estimator.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        """Compute scores on data not used for training."""
        y_cal = np.asarray(y_cal).ravel()
        proba = np.asarray(self.estimator.predict_proba(X_cal), dtype=float)
        if proba.shape[0] != y_cal.shape[0]:
            raise ValueError("X_cal and y_cal have different numbers of rows.")
        classes = np.asarray(self.classes_)
        index = {c: i for i, c in enumerate(classes.tolist())}
        try:
            cols = np.array([index[c] for c in y_cal.tolist()])
        except KeyError as err:
            raise ValueError(
                f"Calibration label {err.args[0]!r} is not among the "
                "estimator's classes_."
            ) from None
        self.calibration_scores_ = 1.0 - proba[np.arange(len(cols)), cols]
        return self

    def predict(self, X):
        """Point predictions from the underlying classifier."""
        return self.estimator.predict(X)

    def predict_set(self, X, alpha: float | None = None):
        """Return prediction sets as a boolean array (n_samples, n_classes).

        Entry ``[i, j]`` is True when ``classes_[j]`` is in the set for
        sample ``i``.
        """
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        q = conformal_quantile(self.calibration_scores_, alpha)
        proba = np.asarray(self.estimator.predict_proba(X), dtype=float)
        return (1.0 - proba) <= q
