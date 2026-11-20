"""Split conformal classification."""

from __future__ import annotations

import numpy as np

from pyconformal._quantile import _check_alpha, conformal_quantile
from pyconformal.scores import get_score


class SplitConformalClassifier:
    """Split (inductive) conformal prediction sets for classification.

    Wraps any classifier with ``fit``, ``predict_proba`` and ``classes_``.
    A class enters the prediction set when its nonconformity score is at
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
    score : {"lac", "aps", "raps", "saps"} or score object, default="lac"
        Nonconformity score; see :mod:`pyconformal.scores`. Pass an object,
        for example ``RAPS(penalty=0.1, k_reg=1)``, to set its parameters.
    class_conditional : bool, default=False
        If True, each class gets its own threshold, calibrated on the
        calibration samples of that class. Coverage then holds within every
        class, not only on average. A class with too few calibration
        samples for ``alpha`` is always included.
    random_state : int, numpy.random.Generator or None
        Seed for randomised scores.
    """

    def __init__(
        self,
        estimator,
        alpha: float = 0.1,
        prefit: bool = False,
        score="lac",
        class_conditional: bool = False,
        random_state=None,
    ):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.prefit = prefit
        self.score = get_score(score)
        self.class_conditional = bool(class_conditional)
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
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

    def _all_scores(self, proba):
        proba = np.asarray(proba, dtype=float)
        u = self._rng.random(proba.shape[0]) if self.score.randomized else None
        return self.score.scores(proba, u)

    def calibrate(self, X_cal, y_cal):
        """Compute scores on data not used for training."""
        y_cal = np.asarray(y_cal).ravel()
        proba = np.asarray(self.estimator.predict_proba(X_cal), dtype=float)
        if proba.shape[0] != y_cal.shape[0]:
            raise ValueError("X_cal and y_cal have different numbers of rows.")
        classes = np.asarray(self.classes_)
        index = {c: i for i, c in enumerate(classes.tolist())}
        try:
            cols = np.array([index[c] for c in y_cal.tolist()], dtype=np.int64)
        except KeyError as err:
            raise ValueError(
                f"Calibration label {err.args[0]!r} is not among the "
                "estimator's classes_."
            ) from None
        self.calibration_scores_ = self._all_scores(proba)[np.arange(len(cols)), cols]
        self._calibration_labels = cols
        return self

    def thresholds(self, alpha: float | None = None):
        """Score threshold, or one per class when ``class_conditional``."""
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() first.")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        s = self.calibration_scores_
        if not self.class_conditional:
            return conformal_quantile(s, alpha)
        q = np.full(len(self.classes_), np.inf)
        for k in range(q.size):
            sk = s[self._calibration_labels == k]
            if sk.size:
                q[k] = conformal_quantile(sk, alpha)
        return q

    def predict(self, X):
        """Point predictions from the underlying classifier."""
        return self.estimator.predict(X)

    def predict_set(self, X, alpha: float | None = None):
        """Return prediction sets as a boolean array (n_samples, n_classes).

        Entry ``[i, j]`` is True when ``classes_[j]`` is in the set for
        sample ``i``.
        """
        q = self.thresholds(alpha)
        proba = np.asarray(self.estimator.predict_proba(X), dtype=float)
        return self._all_scores(proba) <= q
