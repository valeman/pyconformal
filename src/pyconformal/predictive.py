"""Split conformal predictive systems: full predictive distributions for regression.

References
----------
Vovk, Shen, Manokhin and Xie (2019). Nonparametric predictive distributions
based on conformal prediction. Machine Learning 108.

Vovk, Petej, Nouretdinov, Manokhin and Gammerman (2020). Computationally
efficient versions of conformal predictive distributions. Neurocomputing 397.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from pyconformal._base import _Conformal
from pyconformal._quantile import _check_alpha
from pyconformal._search import search

__all__ = ["ConformalPredictiveSystem"]


def _rank(n: int, q: float, side: str) -> int:
    """floor or ceil of q * (n + 1), in exact rational arithmetic."""
    x = (n + 1) * Fraction(repr(float(q)))
    return math.floor(x) if side == "lower" else math.ceil(x)


class _Scores:
    """Sorted calibration scores of one group, with what CRPS needs."""

    def __init__(self, a):
        self.a = np.sort(a)
        n = self.a.size
        self.n = n
        self.prefix = np.concatenate([[0.0], np.cumsum(self.a)])
        # run_end[i]: index just past the run of equal scores containing i
        ends = np.flatnonzero(self.a[1:] != self.a[:-1]) + 1
        ends = np.concatenate([ends, [n]])
        starts = np.concatenate([[0], ends[:-1]])
        self.run_end = np.repeat(ends, ends - starts)
        # Mean absolute difference between two scores (Gini mean difference).
        self.gmd = 2.0 * float(np.dot(2.0 * np.arange(1, n + 1) - n - 1, self.a)) / (n * n)


class ConformalPredictiveSystem(_Conformal):
    """Split conformal predictive system.

    Turns point predictions into predictive distributions. With calibration
    scores ``a_i = (y_i - y_pred_i) / sigma_i`` sorted in increasing order,
    the distribution for a test point has support

        C_i = y_pred + sigma * a_(i),   i = 1 .. n,

    and its distribution function, for a threshold ``y`` and ``tau`` in
    [0, 1], is

        Q(y, tau) = (#{C_i < y} + tau * (#{C_i = y} + 1)) / (n + 1).

    If the calibration and test data are exchangeable and ``tau`` is uniform
    on [0, 1], ``Q(y_true, tau)`` is uniform on [0, 1]: the distributions are
    calibrated in probability. Intervals between a lower ``alpha / 2`` and a
    higher ``1 - alpha / 2`` quantile cover with probability at least
    ``1 - alpha``.

    Three variants, chosen by what is passed to ``calibrate``:

    * standard: neither ``sigma`` nor ``groups``;
    * normalized: ``sigma`` gives a positive difficulty estimate per sample,
      so harder samples get wider distributions;
    * Mondrian: ``groups`` gives a category per sample, and each category is
      calibrated separately.

    Works from predictions (``calibrate(y_pred=..., y=...)``, then
    ``predict_*(y_pred=...)``) or from a model (``ConformalPredictiveSystem(
    model)``, then ``calibrate(X_cal, y_cal)`` and ``predict_*(X)``).

    Parameters
    ----------
    estimator : object or None
        A regressor with ``predict``. None when working from predictions.
    alpha : float, default=0.1
        Default miscoverage level for ``predict_interval``.
    """

    def __init__(self, estimator=None, alpha: float = 0.1):
        self.estimator = estimator
        self.alpha = _check_alpha(alpha)
        self._groups = None

    # ------------------------------------------------------------------ fit

    @staticmethod
    def _sigma(sigma, n, name):
        if sigma is None:
            return None
        s = np.asarray(sigma, dtype=float).ravel()
        if s.shape[0] != n:
            raise ValueError(f"{name} has the wrong length.")
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError(f"{name} must be finite and strictly positive.")
        return s

    def calibrate(self, X=None, y=None, *, y_pred=None, sigma=None, groups=None):
        """Calibrate on data not used to train the model."""
        if y is None:
            raise TypeError("calibrate() needs the outcomes y.")
        sigma_cal, groups_cal = sigma, groups
        y = np.asarray(y, dtype=float).ravel()
        p = self._output(X, y_pred).ravel()
        if y.shape != p.shape:
            raise ValueError("Predictions and y have different lengths.")
        if y.size == 0:
            raise ValueError("The calibration set is empty.")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
            raise ValueError("Predictions and y must be finite.")
        s = self._sigma(sigma_cal, y.size, "sigma")
        a = (y - p) if s is None else (y - p) / s
        self.normalized_ = s is not None
        self.mondrian_ = groups_cal is not None
        if groups_cal is None:
            self._groups = {None: _Scores(a)}
        else:
            g = np.asarray(groups_cal).ravel()
            if g.shape[0] != y.size:
                raise ValueError("groups has the wrong length.")
            self._keys = np.unique(g)
            self._groups = {k: _Scores(a[g == k]) for k in self._keys.tolist()}
        return self

    # ------------------------------------------------------------- plumbing

    def _prepare(self, X, y_pred, sigma, groups):
        if self._groups is None:
            raise RuntimeError("Call calibrate() first.")
        p = self._output(X, y_pred).ravel()
        if self.normalized_ and sigma is None:
            raise ValueError("This system was calibrated with sigma; pass sigma.")
        if not self.normalized_ and sigma is not None:
            raise ValueError("This system was calibrated without sigma; do not pass sigma.")
        s = self._sigma(sigma, p.size, "sigma")
        if self.mondrian_:
            if groups is None:
                raise ValueError("This system was calibrated with groups; pass groups.")
            g = np.asarray(groups).ravel()
            if g.shape[0] != p.size:
                raise ValueError("groups has the wrong length.")
            # Code each sample by its calibration group (one search in a short
            # sorted key list), then bring equal codes together with one
            # stable sort of small integers.
            keys = self._keys
            try:
                codes = np.searchsorted(keys, g)
                seen = keys[np.minimum(codes, keys.shape[0] - 1)] == g
            except TypeError:
                seen = np.zeros(g.shape[0], dtype=bool)
            if not np.all(seen):
                bad = np.asarray(g)[~np.asarray(seen)][0]
                raise ValueError(f"Group {bad!r} was not seen in calibration.")
            small = np.int16 if keys.shape[0] < 32_000 else np.int64
            order = np.argsort(codes.astype(small), kind="stable")
            bounds = np.concatenate([[0], np.cumsum(np.bincount(codes, minlength=keys.shape[0]))])
            parts = [
                (self._groups[k], order[bounds[j] : bounds[j + 1]])
                for j, k in enumerate(keys.tolist())
                if bounds[j + 1] > bounds[j]
            ]
        else:
            if groups is not None:
                raise ValueError("This system was calibrated without groups; do not pass groups.")
            parts = [(self._groups[None], slice(None))]
        return p, (np.ones(p.size) if s is None else s), parts

    # ------------------------------------------------------------ quantiles

    def predict_quantile(self, X=None, q=0.5, *, y_pred=None, sigma=None, groups=None, side="higher"):
        """Quantiles of the predictive distributions.

        ``side="lower"`` returns ``C_(floor(q (n + 1)))`` and ``side="higher"``
        returns ``C_(ceil(q (n + 1)))``, with ``C_(0) = -inf`` and
        ``C_(n + 1) = +inf``. The two differ by at most one calibration score;
        use "lower" for lower bounds and "higher" for upper bounds to keep the
        coverage guarantee.

        Returns an array of shape (n_samples, n_quantiles), or (n_samples,)
        when ``q`` is a single number.
        """
        if side not in ("lower", "higher"):
            raise ValueError("side must be 'lower' or 'higher'.")
        scalar = np.isscalar(q)
        qs = np.atleast_1d(np.asarray(q, dtype=float))
        if np.any(qs < 0) or np.any(qs > 1) or not np.all(np.isfinite(qs)):
            raise ValueError("q must be between 0 and 1.")
        p, s, parts = self._prepare(X, y_pred, sigma, groups)
        out = np.empty((p.size, qs.size))
        for sc, rows in parts:
            ext = np.concatenate([[-np.inf], sc.a, [np.inf]])
            a_q = ext[[min(max(_rank(sc.n, t, side), 0), sc.n + 1) for t in qs]]
            with np.errstate(invalid="ignore"):
                out[rows] = p[rows, None] + s[rows, None] * a_q[None, :]
        return out[:, 0] if scalar else out

    def predict_interval(self, X=None, alpha=None, *, y_pred=None, sigma=None, groups=None):
        """Central prediction intervals of shape (n_samples, 2).

        Covers with probability at least ``1 - alpha``. A bound is infinite
        when the calibration set (of that group) has fewer than
        ``2 / alpha - 1`` points.
        """
        alphas, single = self._alphas(alpha)
        p = self._output(X, y_pred).ravel()  # call the model once
        kw = dict(y_pred=p, sigma=sigma, groups=groups)
        out = np.empty((p.size, 2, len(alphas)))
        out[:, 0, :] = self.predict_quantile(q=[a / 2.0 for a in alphas], side="lower", **kw)
        out[:, 1, :] = self.predict_quantile(q=[1.0 - a / 2.0 for a in alphas], side="higher", **kw)
        return out[:, :, 0] if single else out

    # ------------------------------------------------------------------ cdf

    def predict_cdf(self, X=None, y=None, *, y_pred=None, sigma=None, groups=None, tau=0.5):
        """Value ``Q(y, tau)`` of each predictive distribution at ``y``.

        ``y`` is one threshold per sample, or a single number for all.
        ``tau`` is a number or an array in [0, 1]. Pass uniform random numbers
        to get exactly uniform values at the true outcomes; the default 0.5
        is the deterministic midpoint.
        """
        if y is None:
            raise TypeError("predict_cdf() needs the thresholds y.")
        p, s, parts = self._prepare(X, y_pred, sigma, groups)
        yv = np.broadcast_to(np.asarray(y, dtype=float), p.shape)
        tv = np.broadcast_to(np.asarray(tau, dtype=float), p.shape)
        if np.any(tv < 0) or np.any(tv > 1):
            raise ValueError("tau must be between 0 and 1.")
        z = (yv - p) / s
        out = np.empty(p.size)
        for sc, rows in parts:
            zz = z[rows]
            lt = search(sc.a, zz, "left")
            at = np.minimum(lt, sc.n - 1)
            # number of calibration scores equal to z, from one search
            eq = np.where(sc.a[at] == zz, sc.run_end[at] - lt, 0)
            out[rows] = (lt + tv[rows] * (eq + 1)) / (sc.n + 1)
        return out

    def predict_distribution(self, X=None, *, y_pred=None, sigma=None):
        """Support points ``C_1 <= ... <= C_n`` of each distribution, as an
        array of shape (n_samples, n_calibration). Not available for Mondrian
        systems, whose groups can differ in size."""
        if self._groups is not None and self.mondrian_:
            raise NotImplementedError(
                "predict_distribution is not available for Mondrian systems; "
                "use predict_quantile or predict_cdf."
            )
        p, s, parts = self._prepare(X, y_pred, sigma, None)
        return p[:, None] + s[:, None] * parts[0][0].a[None, :]

    # ----------------------------------------------------------------- crps

    def crps(self, X=None, y=None, *, y_pred=None, sigma=None, groups=None):
        """Continuous ranked probability score per sample (lower is better).

        Scores the distribution that puts mass ``1 / n`` on each support
        point. Computed in closed form from prefix sums, in O(log n) per
        sample, without building the distributions.
        """
        if y is None:
            raise TypeError("crps() needs the outcomes y.")
        p, s, parts = self._prepare(X, y_pred, sigma, groups)
        yt = np.asarray(y, dtype=float).ravel()
        if yt.shape != p.shape:
            raise ValueError("Predictions and y have different lengths.")
        z = (yt - p) / s
        out = np.empty(p.size)
        for sc, rows in parts:
            zz = z[rows]
            k = search(sc.a, zz, "right")
            below = sc.prefix[k]
            # sum over i of |a_i - z|
            abs_sum = zz * k - below + (sc.prefix[-1] - below) - zz * (sc.n - k)
            out[rows] = s[rows] * (abs_sum / sc.n - 0.5 * sc.gmd)
        return out
