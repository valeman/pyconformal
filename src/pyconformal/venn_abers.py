"""Inductive Venn-Abers predictors for binary and multiclass classification.

References
----------
Vovk, Petej and Fedorova (2015). Large-scale probabilistic predictors with
and without guarantees of validity. NeurIPS.

Manokhin (2017). Multi-class probabilistic classification using inductive
and cross Venn-Abers predictors. COPA, PMLR 60.
"""

from __future__ import annotations

import numpy as np

from pyconformal._va_kernel import bridge_slopes

__all__ = ["VennAbers", "MulticlassVennAbers", "VennAbersCalibrator"]

_MERGES = ("log", "brier")


def _merge(p0, p1, merge):
    """Combine the pair (p0, p1) into one probability of the positive class."""
    if merge == "log":
        # Minimax for log loss. p0 <= p1, so the denominator is at least 1.
        return p1 / (1.0 - p0 + p1)
    return p1 + 0.5 * p0 * p0 - 0.5 * p1 * p1  # minimax for Brier loss


def _check_merge(merge):
    if merge not in _MERGES:
        raise ValueError(f"merge must be one of {_MERGES}, got {merge!r}.")
    return merge


def _as_scores(s, name):
    """Accept 1-D scores, or an (n, 2) probability array (column 1 is used)."""
    s = np.asarray(s, dtype=float)
    if s.ndim == 2 and s.shape[1] == 2:
        s = s[:, 1]
    if s.ndim != 1:
        raise ValueError(f"{name} must be 1-D scores or an (n, 2) probability array.")
    if not np.all(np.isfinite(s)):
        raise ValueError(f"{name} must be finite.")
    return s


class VennAbers:
    """Inductive Venn-Abers predictor for binary classification.

    Works on scores, so it fits any model: pass the classifier's score or
    probability of the positive class on a calibration set that was not used
    for training. Any real-valued score works; only its ordering matters.

    For each test score the predictor returns two probabilities,
    ``p0 <= p1``. They are the isotonic-regression fits at the test point
    when it is added to the calibration set with label 0 and with label 1.
    If the calibration and test data are exchangeable, one of the two is
    perfectly calibrated. The width ``p1 - p0`` shows how much the
    calibration set can say about that score.

    Fitting costs O(n log n) for the sort plus a linear-time scan; each
    prediction is a binary search.

    Parameters
    ----------
    merge : {"log", "brier"}, default="log"
        How ``predict_proba`` combines ``(p0, p1)`` into one probability:
        ``p1 / (1 - p0 + p1)`` for log loss, or
        ``p1 + p0**2 / 2 - p1**2 / 2`` for Brier loss.

    Attributes
    ----------
    scores_ : ndarray of shape (k,)
        Sorted distinct calibration scores.
    p0_, p1_ : ndarray of shape (k + 1,)
        Lookup tables used by ``predict_p0_p1``.
    """

    def __init__(self, merge: str = "log"):
        self.merge = _check_merge(merge)
        self.scores_ = None

    def fit(self, scores_cal, y_cal, *, use_numba=None):
        """Calibrate on scores and binary labels.

        ``y_cal`` may be boolean or 0/1. ``use_numba`` forces (True) or
        disables (False) the compiled kernel; by default it is used when
        numba is installed. Both paths give identical results.
        """
        s = _as_scores(scores_cal, "scores_cal")
        y = np.asarray(y_cal).ravel()
        if y.shape[0] != s.shape[0]:
            raise ValueError("scores_cal and y_cal have different lengths.")
        if s.shape[0] == 0:
            raise ValueError("The calibration set is empty.")
        if y.dtype != bool:
            if not np.all((y == 0) | (y == 1)):
                raise ValueError("y_cal must contain only 0/1 or boolean labels.")
        y = y.astype(np.int64)

        c, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
        k = c.shape[0]
        pos = np.bincount(inverse.ravel(), weights=y, minlength=k).astype(np.int64)
        X = np.zeros(k + 1, dtype=np.int64)
        Y = np.zeros(k + 1, dtype=np.int64)
        np.cumsum(counts, out=X[1:])
        np.cumsum(pos, out=Y[1:])

        # Table index i is the group the test point joins:
        #   p1: i = number of calibration scores strictly below the test score
        #       (a label-1 point pools with the next group; above all: 1).
        #   p0: i = number of calibration scores at or below the test score
        #       (a label-0 point pools with the previous group; below all: 0).
        p1 = np.ones(k + 1)
        p1[:k] = bridge_slopes(X, Y, 1, use_numba)
        p0 = np.zeros(k + 1)
        p0[1:] = bridge_slopes(X, Y, 0, use_numba)

        self.scores_, self.p0_, self.p1_ = c, p0, p1
        self.n_calibration_ = int(s.shape[0])
        return self

    def _check_fitted(self):
        if self.scores_ is None:
            raise RuntimeError("Call fit() first.")

    def predict_p0_p1(self, scores):
        """Return the pair (p0, p1) per sample as an array of shape (n, 2)."""
        self._check_fitted()
        s = _as_scores(scores, "scores")
        p0 = self.p0_[np.searchsorted(self.scores_, s, side="right")]
        p1 = self.p1_[np.searchsorted(self.scores_, s, side="left")]
        return np.column_stack([p0, p1])

    def predict_proba(self, scores):
        """Return calibrated probabilities, shape (n, 2): [P(0), P(1)]."""
        pp = self.predict_p0_p1(scores)
        p = _merge(pp[:, 0], pp[:, 1], self.merge)
        return np.column_stack([1.0 - p, p])


class MulticlassVennAbers:
    """Venn-Abers calibration of multiclass probabilities.

    Reduces the problem to binary Venn-Abers predictors and recombines them.

    Parameters
    ----------
    strategy : {"one_vs_one", "one_vs_rest"}, default="one_vs_one"
        ``"one_vs_one"`` fits one binary predictor per pair of classes
        ``(i, j)`` on the calibration rows of those two classes, using the
        score ``p_j / (p_i + p_j)``, and combines the pairwise probabilities
        by pairwise coupling, ``p_i = 1 / (sum_j 1/p_ij - (K - 2))``
        (Manokhin, 2017). ``"one_vs_rest"`` fits one predictor per class on
        its probability column. Both renormalise rows to sum to 1.
    merge : {"log", "brier"}, default="log"
        See :class:`VennAbers`.

    Notes
    -----
    The validity guarantee is that of each binary predictor. The combined
    multiclass probabilities are a heuristic recombination.
    """

    def __init__(self, strategy: str = "one_vs_one", merge: str = "log"):
        if strategy not in ("one_vs_one", "one_vs_rest"):
            raise ValueError("strategy must be 'one_vs_one' or 'one_vs_rest'.")
        self.strategy = strategy
        self.merge = _check_merge(merge)
        self.predictors_ = None

    @staticmethod
    def _pair_score(P, i, j):
        den = P[:, i] + P[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = P[:, j] / den
        return np.where(den > 0, s, 0.5)

    def fit(self, proba_cal, y_cal, classes=None, *, use_numba=None):
        """Calibrate on an (n, K) probability array and labels.

        ``classes`` gives the label of each column; by default the sorted
        distinct labels in ``y_cal``. Every class must occur in ``y_cal``.
        """
        P = np.asarray(proba_cal, dtype=float)
        y = np.asarray(y_cal).ravel()
        if P.ndim != 2 or P.shape[1] < 2:
            raise ValueError("proba_cal must have shape (n, K) with K >= 2.")
        if P.shape[0] != y.shape[0]:
            raise ValueError("proba_cal and y_cal have different lengths.")
        if not np.all(np.isfinite(P)):
            raise ValueError("proba_cal must be finite.")
        classes = np.unique(y) if classes is None else np.asarray(classes)
        K = P.shape[1]
        if classes.shape[0] != K:
            raise ValueError(
                f"proba_cal has {K} columns but there are {classes.shape[0]} classes."
            )
        index = {c: i for i, c in enumerate(classes.tolist())}
        try:
            col = np.array([index[v] for v in y.tolist()], dtype=np.int64)
        except KeyError as err:
            raise ValueError(f"Label {err.args[0]!r} is not in classes.") from None
        missing = np.setdiff1d(np.arange(K), col)
        if missing.size:
            raise ValueError(
                f"Classes {classes[missing].tolist()} do not occur in y_cal."
            )

        self.classes_ = classes
        self.predictors_ = []
        if self.strategy == "one_vs_one":
            self.pairs_ = [(i, j) for i in range(K - 1) for j in range(i + 1, K)]
            for i, j in self.pairs_:
                rows = (col == i) | (col == j)
                va = VennAbers(self.merge).fit(
                    self._pair_score(P[rows], i, j), col[rows] == j, use_numba=use_numba
                )
                self.predictors_.append(va)
        else:
            for i in range(K):
                va = VennAbers(self.merge).fit(P[:, i], col == i, use_numba=use_numba)
                self.predictors_.append(va)
        return self

    def predict_proba(self, proba):
        """Return calibrated probabilities of shape (n, K)."""
        if self.predictors_ is None:
            raise RuntimeError("Call fit() first.")
        P = np.asarray(proba, dtype=float)
        K = self.classes_.shape[0]
        if P.ndim != 2 or P.shape[1] != K:
            raise ValueError(f"proba must have shape (n, {K}).")
        if not np.all(np.isfinite(P)):
            raise ValueError("proba must be finite.")
        out = np.empty((P.shape[0], K))
        if self.strategy == "one_vs_one":
            inv = np.zeros((P.shape[0], K))  # sum over j of 1 / p_ij
            with np.errstate(divide="ignore"):
                for (i, j), va in zip(self.pairs_, self.predictors_):
                    pj = va.predict_proba(self._pair_score(P, i, j))[:, 1]
                    inv[:, i] += 1.0 / (1.0 - pj)
                    inv[:, j] += 1.0 / pj
                out = 1.0 / (inv - (K - 2))
        else:
            for i, va in enumerate(self.predictors_):
                out[:, i] = va.predict_proba(P[:, i])[:, 1]
        total = out.sum(axis=1, keepdims=True)
        # A row can sum to zero only if every class got probability 0.
        return np.where(total > 0, out / np.where(total > 0, total, 1.0), 1.0 / K)


class VennAbersCalibrator:
    """Venn-Abers calibration around any probabilistic classifier.

    Wraps a classifier with ``fit``, ``predict_proba`` and ``classes_``.
    Binary problems use :class:`VennAbers`; problems with more classes use
    :class:`MulticlassVennAbers`.

    Parameters
    ----------
    estimator : object
        The classifier to calibrate.
    prefit : bool, default=False
        If True, ``estimator`` is already trained; go straight to
        ``calibrate``.
    strategy : {"one_vs_one", "one_vs_rest"}, default="one_vs_one"
        Multiclass reduction; ignored for binary problems.
    merge : {"log", "brier"}, default="log"
        See :class:`VennAbers`.
    """

    def __init__(self, estimator, prefit=False, strategy="one_vs_one", merge="log"):
        self.estimator = estimator
        self.prefit = prefit
        self.strategy = strategy
        self.merge = _check_merge(merge)
        self.calibrator_ = None

    @property
    def classes_(self):
        return np.asarray(self.estimator.classes_)

    def fit(self, X, y):
        """Train the underlying classifier on the proper training set."""
        if self.prefit:
            raise RuntimeError("prefit=True: the estimator is already trained.")
        self.estimator.fit(X, y)
        return self

    def calibrate(self, X_cal, y_cal):
        """Calibrate on data not used for training."""
        P = np.asarray(self.estimator.predict_proba(X_cal), dtype=float)
        y = np.asarray(y_cal).ravel()
        classes = self.classes_
        if classes.shape[0] == 2:
            known = np.isin(y, classes)
            if not np.all(known):
                raise ValueError(
                    f"Label {y[~known][0]!r} is not among the estimator's classes_."
                )
            self.calibrator_ = VennAbers(self.merge).fit(P[:, 1], y == classes[1])
        else:
            self.calibrator_ = MulticlassVennAbers(self.strategy, self.merge).fit(
                P, y, classes=classes
            )
        return self

    def _proba(self, X):
        if self.calibrator_ is None:
            raise RuntimeError("Call calibrate() first.")
        return np.asarray(self.estimator.predict_proba(X), dtype=float)

    def predict_proba(self, X):
        """Calibrated class probabilities, columns ordered as ``classes_``."""
        P = self._proba(X)
        if isinstance(self.calibrator_, VennAbers):
            return self.calibrator_.predict_proba(P[:, 1])
        return self.calibrator_.predict_proba(P)

    def predict_p0_p1(self, X):
        """Binary problems only: the pair (p0, p1) for ``classes_[1]``."""
        P = self._proba(X)
        if not isinstance(self.calibrator_, VennAbers):
            raise RuntimeError("predict_p0_p1 is defined for binary problems only.")
        return self.calibrator_.predict_p0_p1(P[:, 1])

    def predict(self, X):
        """Class with the highest calibrated probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
