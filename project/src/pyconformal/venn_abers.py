"""Inductive Venn-Abers predictors for binary and multiclass classification.

References
----------
Vovk, Petej and Fedorova (2015). Large-scale probabilistic predictors with
and without guarantees of validity. NeurIPS.

Manokhin (2017). Multi-class probabilistic classification using inductive
and cross Venn-Abers predictors. COPA, PMLR 60.
"""

from __future__ import annotations

import copy

import numpy as np

from pyconformal._base import _Conformal, _rows
from pyconformal._va_kernel import _release_hulls, bridge_slopes, lookup

__all__ = ["VennAbers", "CrossVennAbers"]

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


class _IVAP:
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

        # One sort; groups of equal scores end where the sorted value changes.
        order = np.argsort(s)
        ss = s[order]
        ends = np.flatnonzero(ss[1:] != ss[:-1])
        ends = np.concatenate([ends, [ss.shape[0] - 1]])
        c = ss[ends]
        k = c.shape[0]
        X = np.zeros(k + 1, dtype=np.int64)
        Y = np.zeros(k + 1, dtype=np.int64)
        X[1:] = ends + 1
        Y[1:] = np.cumsum(y[order])[ends]

        # Table index i is the group the test point joins:
        #   p1: i = number of calibration scores strictly below the test score
        #       (a label-1 point pools with the next group; above all: 1).
        #   p0: i = number of calibration scores at or below the test score
        #       (a label-0 point pools with the previous group; below all: 0).
        p1 = np.ones(k + 1)
        p1[:k] = bridge_slopes(X, Y, 1, use_numba)
        p0 = np.zeros(k + 1)
        p0[1:] = bridge_slopes(X, Y, 0, use_numba)

        _release_hulls()
        self.scores_, self.p0_, self.p1_ = c, p0, p1
        self.n_calibration_ = int(s.shape[0])
        return self

    def _check_fitted(self):
        if self.scores_ is None:
            raise RuntimeError("Call fit() first.")

    def _pair(self, scores):
        self._check_fitted()
        return lookup(self.scores_, self.p0_, self.p1_, _as_scores(scores, "scores"))

    def predict_p0_p1(self, scores):
        """Return the pair (p0, p1) per sample as an array of shape (n, 2)."""
        p0, p1 = self._pair(scores)
        out = np.empty((2, p0.shape[0]))
        out[0] = p0
        out[1] = p1
        return out.T

    def predict_positive(self, scores):
        """Calibrated probability of the positive class, shape (n,)."""
        p0, p1 = self._pair(scores)
        return _merge(p0, p1, self.merge)

    def predict_proba(self, scores):
        """Return calibrated probabilities, shape (n, 2): [P(0), P(1)]."""
        p = self.predict_positive(scores)
        out = np.empty((2, p.shape[0]))
        np.subtract(1.0, p, out=out[0])
        out[1] = p
        return out.T


class _MulticlassIVAP:
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
        See the binary calibrator.

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
                va = _IVAP(self.merge).fit(
                    self._pair_score(P[rows], i, j), col[rows] == j, use_numba=use_numba
                )
                self.predictors_.append(va)
        else:
            for i in range(K):
                va = _IVAP(self.merge).fit(P[:, i], col == i, use_numba=use_numba)
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
                    pj = va.predict_positive(self._pair_score(P, i, j))
                    inv[:, i] += 1.0 / (1.0 - pj)
                    inv[:, j] += 1.0 / pj
                out = 1.0 / (inv - (K - 2))
        else:
            for i, va in enumerate(self.predictors_):
                out[:, i] = va.predict_positive(P[:, i])
        total = out.sum(axis=1, keepdims=True)
        # A row can sum to zero only if every class got probability 0.
        return np.where(total > 0, out / np.where(total > 0, total, 1.0), 1.0 / K)


class VennAbers(_Conformal):
    _calibration_attrs = ("calibrator_", "_n_classes")
    """Venn-Abers calibration of class probabilities, binary and multiclass.

    Turns the scores of any classifier into calibrated probabilities. For
    binary problems each sample also gets the pair ``(p0, p1)``, whose width
    shows how uncertain the calibrated probability is.

    Works from model outputs::

        va = VennAbers()
        va.calibrate(proba=proba_cal, y=y_cal)      # or scores=, for binary
        va.predict_proba(proba=proba_test)

    or from a model::

        va = VennAbers(model)
        va.fit(X_train, y_train)                     # skip if already trained
        va.calibrate(X_cal, y_cal)
        va.predict_proba(X_test)

    Parameters
    ----------
    estimator : object or None
        A classifier with ``predict_proba`` and ``classes_``. None when
        working from model outputs.
    strategy : {"one_vs_one", "one_vs_rest"}, default="one_vs_one"
        Multiclass reduction; ignored for binary problems.
    merge : {"log", "brier"}, default="log"
        How ``p0`` and ``p1`` are merged into one probability.
    classes : array-like or None
        Class label of each probability column, when there is no estimator.
        Default: 0, 1, ..., K - 1.
    """

    _model_method = "predict_proba"
    _given = "proba"

    def __init__(self, estimator=None, strategy="one_vs_one", merge="log", classes=None):
        self.estimator = estimator
        self.strategy = strategy
        self.merge = _check_merge(merge)
        self.classes = classes
        self.calibrator_ = None
        self._n_classes = None

    @property
    def classes_(self):
        if self.classes is not None:
            return np.asarray(self.classes)
        if self.estimator is not None:
            return np.asarray(self.estimator.classes_)
        if self._n_classes is None:
            raise RuntimeError("Call calibrate() first.")
        return np.arange(self._n_classes)

    def _inputs(self, X, proba, scores):
        """Binary scores of shape (n,), or probabilities of shape (n, K)."""
        if scores is not None:
            if X is not None or proba is not None:
                raise TypeError("Pass one of X, proba= or scores=.")
            return np.asarray(scores, dtype=float).ravel(), True
        P = self._output(X, proba)
        if P.ndim != 2 or P.shape[1] < 2:
            raise ValueError(
                "Probabilities must have shape (n_samples, n_classes); "
                "for one score per sample use scores=."
            )
        return (P[:, 1], True) if P.shape[1] == 2 else (P, False)

    def calibrate(self, X=None, y=None, *, proba=None, scores=None, use_numba=None):
        """Calibrate on data not used for training."""
        if y is None:
            raise TypeError("calibrate() needs the labels y.")
        Z, binary = self._inputs(X, proba, scores)
        y = np.asarray(y).ravel()
        self._n_classes = 2 if binary else Z.shape[1]
        classes = self.classes_
        if classes.shape[0] != self._n_classes:
            raise ValueError("The number of classes does not match the model outputs.")
        if binary:
            # the positive class is the second of ``classes_``, whatever the
            # label type (bool labels included)
            known = np.isin(y, classes)
            if not np.all(known):
                raise ValueError(f"Label {y[~known][0]!r} is not among the classes.")
            positive = y == classes[1]
            self.calibrator_ = _IVAP(self.merge).fit(Z, positive, use_numba=use_numba)
        else:
            self.calibrator_ = _MulticlassIVAP(self.strategy, self.merge).fit(
                Z, y, classes=classes, use_numba=use_numba
            )
        self._binary = binary
        return self

    def _check(self):
        if self.calibrator_ is None:
            raise RuntimeError("Call calibrate() first.")

    def predict_proba(self, X=None, *, proba=None, scores=None):
        """Calibrated class probabilities, columns ordered as ``classes_``."""
        self._check()
        Z, binary = self._inputs(X, proba, scores)
        if binary != self._binary:
            raise ValueError("The number of classes differs from calibration.")
        return self.calibrator_.predict_proba(Z)

    def predict_p0_p1(self, X=None, *, proba=None, scores=None):
        """Binary problems only: the pair (p0, p1) for ``classes_[1]``,
        shape (n_samples, 2)."""
        self._check()
        if not self._binary:
            raise RuntimeError("predict_p0_p1 is defined for binary problems only.")
        Z, _ = self._inputs(X, proba, scores)
        return self.calibrator_.predict_p0_p1(Z)

    def predict(self, X=None, *, proba=None, scores=None):
        """Class with the highest calibrated probability."""
        return self.classes_[np.argmax(self.predict_proba(X, proba=proba, scores=scores), axis=1)]


class CrossVennAbers(_Conformal):
    """Cross Venn-Abers calibration (CVAP; Vovk and Petej, 2014) for binary
    classification.

    Inductive Venn-Abers sets data aside for calibration. The cross version
    uses every row: the data are cut into K folds, a model is trained
    without each fold and calibrated on it, and the K calibrated answers for
    a new sample are merged. Probabilities are usually a little sharper
    than with one split, at the cost of training K models.

    From a model::

        cva = CrossVennAbers(classifier, cv=5)
        cva.fit(X, y)                      # trains and calibrates
        cva.predict_proba(X_test)

    From scores, when the fold models were trained elsewhere::

        cva = CrossVennAbers()
        cva.calibrate(scores=oof_scores, y=y, folds=fold_of_each_row)
        cva.predict_proba(scores=test_scores)     # (n_test, K), column k from model k

    Parameters
    ----------
    estimator : object or None
        A binary classifier with ``fit`` and ``predict_proba``; it is copied
        once per fold.
    cv : int, default=5
    merge : {"log", "brier"}, default="log"
        How the K pairs (p0, p1) become one probability: geometric means
        (best for log loss) or the Brier-optimal formula.
    shuffle : bool, default=True
    random_state : int or None

    Reference
    ---------
    Vovk and Petej (2014). Venn-Abers predictors. UAI.
    Vovk, Petej and Fedorova (2015). Large-scale probabilistic predictors
    with and without guarantees of validity. NeurIPS.
    """

    _model_method = "predict_proba"
    _given = "scores"

    def __init__(self, estimator=None, cv: int = 5, merge: str = "log", shuffle: bool = True, random_state=None):
        self.estimator = estimator
        self.cv = cv
        self.merge = _check_merge(merge)
        self.shuffle = shuffle
        self.random_state = random_state
        self.calibrators_ = None
        self.estimators_ = None
        self.classes_ = None

    def fit(self, X, y, **fit_params):
        """Train one copy of the model without each fold and calibrate it on
        that fold. No separate ``calibrate`` call is needed."""
        if self.estimator is None:
            raise TypeError("fit() needs an estimator. With scores, use calibrate().")
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("CrossVennAbers handles two classes.")
        n, K = y.shape[0], int(self.cv)
        if not 2 <= K <= n:
            raise ValueError("cv must be between 2 and the number of rows.")
        order = np.random.default_rng(self.random_state).permutation(n) if self.shuffle else np.arange(n)
        folds = np.empty(n, dtype=np.int64)
        folds[order] = np.arange(n) % K
        oof = np.empty(n)
        models = []
        for k in range(K):
            test = folds == k
            m = copy.deepcopy(self.estimator).fit(X[~test], y[~test], **_rows(fit_params, ~test, len(y)))
            if not np.array_equal(np.asarray(m.classes_), classes):
                raise ValueError("A training fold is missing one of the classes; use fewer folds.")
            oof[test] = np.asarray(m.predict_proba(X[test]), dtype=float)[:, 1]
            models.append(m)
        self.estimators_ = models
        return self.calibrate(scores=oof, y=y, folds=folds, classes=classes)

    def calibrate(self, X=None, y=None, *, scores=None, folds=None, classes=None, use_numba=None):
        """Calibrate from out-of-fold scores: ``scores[i]`` must come from
        the model trained without row ``i``, and ``folds[i]`` says which
        model that is (0 .. K - 1). ``y`` is boolean, or holds two labels
        (the larger one is the positive class unless ``classes`` is given).
        """
        if X is not None:
            raise TypeError("Cross Venn-Abers needs out-of-fold scores; with a model, call fit(X, y).")
        if y is None or scores is None or folds is None:
            raise TypeError("calibrate() needs scores=, y= and folds=.")
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(y).ravel()
        f = np.asarray(folds).ravel()
        if not (s.shape == y.shape == f.shape):
            raise ValueError("scores, y and folds have different lengths.")
        if classes is None:
            classes = np.array([False, True]) if y.dtype == bool else np.unique(y)
        classes = np.asarray(classes)
        if classes.shape[0] != 2 or not np.all(np.isin(y, classes)):
            raise ValueError("y must hold exactly two classes.")
        keys = np.unique(f)
        if keys.shape[0] < 2 or not np.array_equal(keys, np.arange(keys.shape[0])):
            raise ValueError("folds must use every label 0 .. K - 1, with K at least 2.")
        positive = y == classes[1]
        self.calibrators_ = [_IVAP(self.merge).fit(s[f == k], positive[f == k], use_numba=use_numba) for k in keys]
        self.classes_ = classes
        return self

    def _fold_scores(self, X, scores):
        if self.calibrators_ is None:
            raise RuntimeError("Call fit() or calibrate() first.")
        K = len(self.calibrators_)
        if scores is not None:
            if X is not None:
                raise TypeError("Pass X or scores=, not both.")
            S = np.asarray(scores, dtype=float)
        else:
            if X is None or self.estimators_ is None:
                raise TypeError("Pass X (after fit) or scores= with one column per fold model.")
            S = np.column_stack([np.asarray(m.predict_proba(X), dtype=float)[:, 1] for m in self.estimators_])
        if S.ndim != 2 or S.shape[1] != K:
            raise ValueError(f"Expected scores of shape (n_samples, {K}): one column per fold model.")
        return S

    def predict_pairs(self, X=None, *, scores=None):
        """The Venn-Abers pair from every fold, shape (n_samples, K, 2):
        ``[..., 0]`` is p0 and ``[..., 1]`` is p1."""
        S = self._fold_scores(X, scores)
        return np.stack([c.predict_p0_p1(S[:, k]) for k, c in enumerate(self.calibrators_)], axis=1)

    def predict_proba(self, X=None, *, scores=None):
        """Calibrated probabilities, shape (n_samples, 2), columns ordered
        as ``classes_``."""
        P = self.predict_pairs(X, scores=scores)
        p0, p1 = P[:, :, 0], P[:, :, 1]
        if self.merge == "log":
            with np.errstate(divide="ignore"):
                up = np.exp(np.mean(np.log(p1), axis=1))
                down = np.exp(np.mean(np.log(1.0 - p0), axis=1))
            tot = up + down
            p = np.where(tot > 0, up / np.where(tot > 0, tot, 1.0), 0.5)
        else:
            p = np.mean(p1 + 0.5 * p0 * p0 - 0.5 * p1 * p1, axis=1)
        out = np.empty((p.shape[0], 2))
        out[:, 0] = 1.0 - p
        out[:, 1] = p
        return out

    def predict(self, X=None, *, scores=None):
        """Class with the higher calibrated probability."""
        return self.classes_[np.argmax(self.predict_proba(X, scores=scores), axis=1)]

