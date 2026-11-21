"""Split conformal classification."""

from __future__ import annotations

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._quantile import _check_alpha, conformal_quantile
from pyconformal.scores import get_score


class SplitConformalClassifier(_Conformal):
    """Split (inductive) conformal prediction sets for classification.

    A class enters the prediction set when its nonconformity score is at
    most the finite-sample conformal quantile of the calibration scores.

    If the calibration and test data are exchangeable, the prediction set
    contains the true label with probability at least ``1 - alpha``
    (marginal coverage). Sets can be empty or contain several classes.

    Works from probabilities::

        cp = SplitConformalClassifier(alpha=0.1, score="aps")
        cp.calibrate(proba=proba_cal, y=y_cal)
        cp.predict_set(proba=proba_test)

    or from a model::

        cp = SplitConformalClassifier(model, alpha=0.1, score="aps")
        cp.fit(X_train, y_train)          # skip if already trained
        cp.calibrate(X_cal, y_cal)
        cp.predict_set(X_test)

    Parameters
    ----------
    estimator : object or None
        A classifier with ``predict_proba`` and ``classes_`` (and ``fit``, if
        it is to be trained here). None when working from probabilities.
    alpha : float, default=0.1
        Default miscoverage level; 0.1 gives 90% prediction sets.
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
    classes : array-like or None
        Class label of each probability column, when there is no estimator.
        Default: 0, 1, ..., K - 1.
    """

    _model_method = "predict_proba"
    _given = "proba"

    def __init__(
        self,
        estimator=None,
        alpha: float = 0.1,
        score="lac",
        class_conditional: bool = False,
        random_state=None,
        classes=None,
    ):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.score = get_score(score)
        self.class_conditional = bool(class_conditional)
        self.random_state = random_state
        self.classes = classes
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self._n_classes = None
        self.calibration_scores_ = None

    @property
    def classes_(self):
        if self.classes is not None:
            return np.asarray(self.classes)
        if self.estimator is not None:
            return np.asarray(self.estimator.classes_)
        if self._n_classes is None:
            raise RuntimeError("Call calibrate() first.")
        return np.arange(self._n_classes)

    def _draw_u(self, n):
        return self._rng.random(n) if self.score.randomized else None

    def _proba(self, X, proba):
        P = self._output(X, proba)
        if P.ndim != 2:
            raise ValueError("Probabilities must have shape (n_samples, n_classes).")
        return P

    def calibrate(self, X=None, y=None, *, proba=None):
        """Compute scores on data not used for training."""
        if y is None:
            raise TypeError("calibrate() needs the labels y.")
        y = np.asarray(y).ravel()
        P = self._proba(X, proba)
        if P.shape[0] != y.shape[0]:
            raise ValueError("Probabilities and y have different numbers of rows.")
        self._n_classes = P.shape[1]
        classes = self.classes_
        if classes.shape[0] != P.shape[1]:
            raise ValueError("The number of classes does not match the probability columns.")
        index = {c: i for i, c in enumerate(classes.tolist())}
        try:
            cols = np.array([index[c] for c in y.tolist()], dtype=np.int64)
        except KeyError as err:
            raise ValueError(
                f"Calibration label {err.args[0]!r} is not among the classes."
            ) from None
        self.calibration_scores_ = self.score.label_scores(P, cols, self._draw_u(len(cols)))
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

    def predict(self, X=None, *, proba=None):
        """Most likely class."""
        return self.classes_[np.argmax(self._proba(X, proba), axis=1)]

    def predict_set(self, X=None, alpha=None, *, proba=None):
        """Prediction sets as a boolean array (n_samples, n_classes).

        Entry ``[i, j]`` is True when ``classes_[j]`` is in the set for
        sample ``i``. ``alpha`` overrides the default level without
        recalibrating; a list of levels gives shape
        (n_samples, n_classes, n_levels), with the sets nested.
        """
        alphas, single = self._alphas(alpha)
        qs = [self.thresholds(a) for a in alphas]
        P = self._proba(X, proba)
        u = self._draw_u(P.shape[0])  # one draw, so the sets for several levels are nested
        if single:
            return self.score.sets(P, qs[0], u)
        return np.stack([self.score.sets(P, q, u) for q in qs], axis=2)
