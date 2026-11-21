"""EnbPI and SPCI: prediction intervals for time series from a bootstrap
ensemble, with no data set aside for calibration.

Both train ``B`` copies of the model on bootstrap samples of the training
data. Each training point is then predicted only by the copies that did not
see it, which gives honest residuals for every training point. As new
observations arrive, their residuals replace the oldest ones.

* EnbPI (Xu and Xie, 2021) uses the ``1 - alpha`` quantile of the recent
  absolute residuals as the half-width.
* SPCI (Xu and Xie, 2023) predicts the quantiles of the next residual from
  the residuals just before it, with a quantile regression forest, so the
  interval reacts to runs of large or one-sided errors.

Coverage for both is approximate (it holds as the series grows, under
conditions on the errors), not a finite-sample guarantee.

References
----------
Xu and Xie (2021). Conformal prediction interval for dynamic time-series.
ICML.

Xu and Xie (2023). Sequential predictive conformal inference for time
series. ICML.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._online_kernel import aci_replay
from pyconformal._quantile import _check_alpha, conformal_quantile

__all__ = ["EnbPI", "SPCI", "EnsembleResult"]


@dataclass
class EnsembleResult:
    """Result of replaying a stream: ``intervals`` (T, 2), ``predictions``
    (T,), ``errors`` (T,) and the overall ``coverage``."""

    intervals: np.ndarray
    predictions: np.ndarray
    errors: np.ndarray

    @property
    def coverage(self) -> float:
        return float(1.0 - self.errors.mean()) if self.errors.size else float("nan")


class EnbPI(_Conformal):
    """Ensemble batch prediction intervals (EnbPI).

    From a model::

        enb = EnbPI(model, alpha=0.1, n_bootstraps=30)
        enb.fit(X_train, y_train)            # trains the ensemble and calibrates
        result = enb.run(X_test, y_test)     # one step at a time, residuals sliding
        result.intervals

    or step by step with ``predict_interval(X_next)`` and
    ``update(X_next, y_next)``.

    From predictions, when the ensemble was trained elsewhere: ``calibrate(
    y_pred=member_predictions, y=y, in_bag=in_bag)`` with one column per
    ensemble member, then pass member predictions (n, B) as ``y_pred=``.

    Parameters
    ----------
    estimator : object or None
        A regressor with ``fit`` and ``predict``; copied once per member.
    alpha : float, default=0.1
    n_bootstraps : int, default=30
    batch_size : int, default=1
        In ``run``, residuals are refreshed after every ``batch_size``
        observations.
    random_state : int or None

    The centre of the interval is the mean of all members. The half-width is
    the finite-sample conformal quantile of the residual window, which is
    never narrower than the plain ``1 - alpha`` quantile used in the paper.
    """

    _signed = False

    def __init__(self, estimator=None, alpha: float = 0.1, n_bootstraps: int = 30, batch_size: int = 1, random_state=None):
        if int(n_bootstraps) < 2 or int(batch_size) < 1:
            raise ValueError("n_bootstraps must be at least 2 and batch_size at least 1.")
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self.n_bootstraps = int(n_bootstraps)
        self.batch_size = int(batch_size)
        self.random_state = random_state
        self.estimators_ = None
        self.residuals_ = None

    # ------------------------------------------------------------- training

    def fit(self, X, y, **fit_params):
        """Train the bootstrap ensemble and compute out-of-bag residuals.
        No separate ``calibrate`` call is needed."""
        if self.estimator is None:
            raise TypeError("fit() needs an estimator. With member predictions, use calibrate().")
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        n, B = y.shape[0], self.n_bootstraps
        rng = np.random.default_rng(self.random_state)
        in_bag = np.zeros((n, B), dtype=bool)
        P = np.empty((n, B))
        models = []
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            in_bag[idx, b] = True
            m = copy.deepcopy(self.estimator).fit(X[idx], y[idx], **fit_params)
            P[:, b] = np.asarray(m.predict(X), dtype=float).ravel()
            models.append(m)
        self.estimators_ = models
        return self.calibrate(y_pred=P, y=y, in_bag=in_bag)

    def calibrate(self, X=None, y=None, *, y_pred=None, in_bag=None):
        """Residuals from member predictions on the training rows:
        ``y_pred[i, b]`` is member ``b``'s prediction for row ``i`` and
        ``in_bag[i, b]`` says whether member ``b`` was trained on row ``i``.
        """
        if X is not None:
            raise TypeError("With a model, call fit(X, y).")
        if y is None or y_pred is None or in_bag is None:
            raise TypeError("calibrate() needs y_pred=, y= and in_bag=.")
        y = np.asarray(y, dtype=float).ravel()
        P = np.asarray(y_pred, dtype=float)
        bag = np.asarray(in_bag, dtype=bool)
        if P.ndim != 2 or P.shape != bag.shape or P.shape[0] != y.shape[0]:
            raise ValueError("y_pred and in_bag must have shape (n_samples, n_members).")
        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(y))):
            raise ValueError("y_pred and y must be finite.")
        out = ~bag
        count = out.sum(axis=1)
        loo = np.where(count > 0, (P * out).sum(axis=1) / np.maximum(count, 1), P.mean(axis=1))
        self.n_members_ = P.shape[1]
        self.n_starved_ = int((count == 0).sum())  # rows every member saw: predicted by all members
        r = y - loo
        self.residuals_ = r if self._signed else np.abs(r)
        self._after_calibrate()
        return self

    def _after_calibrate(self):
        pass

    # ----------------------------------------------------------- prediction

    def _members(self, X, y_pred):
        if self.residuals_ is None:
            raise RuntimeError("Call fit() or calibrate() first.")
        if y_pred is not None:
            if X is not None:
                raise TypeError("Pass X or y_pred=, not both.")
            P = np.asarray(y_pred, dtype=float)
        else:
            if X is None or self.estimators_ is None:
                raise TypeError("Pass X (after fit) or y_pred= with one column per member.")
            # predict_interval(X) followed by update(X, y) asks for the same
            # rows twice; the members are only run once
            Xa = np.asarray(X)
            last = getattr(self, "_last_members", None)
            if last is not None and last[0] is self.estimators_ and Xa.size <= 4096 and Xa.shape == last[1].shape and np.array_equal(Xa, last[1]):
                P = last[2]
            else:
                P = np.column_stack([np.asarray(m.predict(X), dtype=float).ravel() for m in self.estimators_])
                if Xa.size <= 4096 and Xa.dtype != object:
                    self._last_members = (self.estimators_, Xa.copy(), P)
        if P.ndim != 2 or P.shape[1] != self.n_members_:
            raise ValueError(f"Expected predictions of shape (n_samples, {self.n_members_}).")
        return P

    def predict(self, X=None, *, y_pred=None):
        """Point prediction: the mean of the ensemble."""
        return self._members(X, y_pred).mean(axis=1)

    def predict_interval(self, X=None, alpha=None, *, y_pred=None):
        """Intervals from the current residuals, shape (n_samples, 2), or
        (n_samples, 2, n_levels) for a list of levels."""
        alphas, single = self._alphas(alpha)
        c = self.predict(X, y_pred=y_pred)
        out = np.empty((c.shape[0], 2, len(alphas)))
        for j, a in enumerate(alphas):
            q = conformal_quantile(self.residuals_, a)
            out[:, 0, j], out[:, 1, j] = c - q, c + q
        return out[:, :, 0] if single else out

    def update(self, X=None, y=None, *, y_pred=None):
        """Add the residuals of newly observed outcomes and drop as many of
        the oldest ones."""
        if y is None:
            raise TypeError("update() needs the outcomes y.")
        c = self.predict(X, y_pred=y_pred)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape != c.shape:
            raise ValueError("Predictions and y have different lengths.")
        r = y - c
        new = r if self._signed else np.abs(r)
        self.residuals_ = np.concatenate([self.residuals_, new])[new.shape[0] :]
        return self

    def run(self, X=None, y=None, *, y_pred=None) -> EnsembleResult:
        """Predict a stream one step at a time, refreshing the residuals
        after every ``batch_size`` observations. Leaves the object as if
        ``update`` had been called for the whole stream."""
        if y is None:
            raise TypeError("run() needs the outcomes y.")
        c = self.predict(X, y_pred=y_pred)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape != c.shape:
            raise ValueError("Predictions and y have different lengths.")
        s = np.abs(y - c)
        # a sliding-window conformal quantile is ACI with step size 0
        q = aci_replay(self.residuals_, s, self.alpha, 0.0, self.alpha, self.residuals_.shape[0])[0]
        if self.batch_size > 1:
            q = q[(np.arange(q.shape[0]) // self.batch_size) * self.batch_size]
        self.residuals_ = np.concatenate([self.residuals_, s])[s.shape[0] :]
        return EnsembleResult(np.column_stack([c - q, c + q]), c, s > q)


class SPCI(EnbPI):
    """Sequential predictive conformal inference (SPCI).

    Same ensemble and out-of-bag residuals as :class:`EnbPI`, but signed.
    To predict step ``t``, a quantile regression forest is fitted on pairs
    (the ``past_window`` residuals before a point, the residual at that
    point) and asked for the quantiles of the next residual given the latest
    window. The interval is ``[centre + Q(beta), centre + Q(1 - alpha +
    beta)]``, with ``beta`` in ``[0, alpha]`` chosen to make it narrowest,
    so it can be asymmetric.

    Parameters
    ----------
    estimator, alpha, n_bootstraps, random_state : as in EnbPI
    past_window : int or None
        Residuals used as inputs. Default: a tenth of the training
        residuals, between 5 and 100.
    beta_bins : int, default=5
        Values of ``beta`` tried.
    refit_every : int, default=1
        Refit the forest after this many new observations. 1 follows the
        paper; larger values are faster.
    n_estimators : int, default=20
        Trees in the quantile forest.
    min_samples_leaf : int, default=5

    Uses :class:`pyconformal.QuantileForest`, which needs numba.
    """

    _signed = True

    def __init__(self, estimator=None, alpha: float = 0.1, n_bootstraps: int = 30, past_window=None,
                 beta_bins: int = 5, refit_every: int = 1, n_estimators: int = 20, min_samples_leaf: int = 5,
                 random_state=None):
        super().__init__(estimator, alpha, n_bootstraps, 1, random_state)
        if int(beta_bins) < 1 or int(refit_every) < 1:
            raise ValueError("beta_bins and refit_every must be at least 1.")
        self.past_window = past_window
        self.beta_bins = int(beta_bins)
        self.refit_every = int(refit_every)
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in (
            "estimator", "alpha", "n_bootstraps", "past_window", "beta_bins", "refit_every", "n_estimators",
            "min_samples_leaf", "random_state")}

    def _after_calibrate(self):
        n = self.residuals_.shape[0]
        w = min(100, max(5, n // 10)) if self.past_window is None else int(self.past_window)
        if n < w + 2 * self.min_samples_leaf:
            raise ValueError("Too few training residuals for this past_window.")
        self.past_window_ = w
        self._forest = None
        self._since_fit = 0

    def _fit_forest(self):
        from pyconformal.forest import QuantileForest

        r, w = self.residuals_, self.past_window_
        windows = np.lib.stride_tricks.sliding_window_view(r, w)[:-1]  # row i: r[i : i + w]
        self._forest = QuantileForest(
            self.n_estimators, min_samples_leaf=self.min_samples_leaf, max_features=1.0 / 3.0,
            random_state=self.random_state,
        ).fit(windows, r[w:])
        self._since_fit = 0

    def _bounds(self, alpha):
        """(lower, upper) offsets for the next step at level ``alpha``."""
        if self._forest is None or self._since_fit >= self.refit_every:
            self._fit_forest()
        betas = np.linspace(0.0, alpha, self.beta_bins + 1)[:-1] if self.beta_bins > 1 else np.array([alpha / 2])
        levels = np.concatenate([betas, 1.0 - alpha + betas])
        q = self._forest.predict(self.residuals_[-self.past_window_ :][None, :], quantiles=levels)[0]
        lo, hi = q[: betas.size], q[betas.size :]
        k = int(np.argmin(hi - lo))
        return lo[k], hi[k]

    def predict_interval(self, X=None, alpha=None, *, y_pred=None):
        """Interval for the next step. Every row passed is treated as the
        next step, with the residuals known now; for a stream use ``run``.
        """
        alphas, single = self._alphas(alpha)
        c = self.predict(X, y_pred=y_pred)
        out = np.empty((c.shape[0], 2, len(alphas)))
        for j, a in enumerate(alphas):
            lo, hi = self._bounds(a)
            out[:, 0, j], out[:, 1, j] = c + lo, c + hi
        return out[:, :, 0] if single else out

    def update(self, X=None, y=None, *, y_pred=None):
        n_before = self.residuals_.shape[0]
        super().update(X, y, y_pred=y_pred)
        self._since_fit += int(np.asarray(y).size)
        assert self.residuals_.shape[0] == n_before
        return self

    def run(self, X=None, y=None, *, y_pred=None) -> EnsembleResult:
        """Predict a stream one step at a time, adding each new residual
        before the next step."""
        if y is None:
            raise TypeError("run() needs the outcomes y.")
        c = self.predict(X, y_pred=y_pred)
        y = np.asarray(y, dtype=float).ravel()
        if y.shape != c.shape:
            raise ValueError("Predictions and y have different lengths.")
        iv = np.empty((c.shape[0], 2))
        for t in range(c.shape[0]):
            lo, hi = self._bounds(self.alpha)
            iv[t] = c[t] + lo, c[t] + hi
            self.residuals_ = np.append(self.residuals_[1:], y[t] - c[t])
            self._since_fit += 1
        return EnsembleResult(iv, c, (y < iv[:, 0]) | (y > iv[:, 1]))
