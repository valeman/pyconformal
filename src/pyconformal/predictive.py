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
        # Mean absolute difference between two scores (Gini mean difference).
        self.gmd = 2.0 * float(np.dot(2.0 * np.arange(1, n + 1) - n - 1, self.a)) / (n * n)


class ConformalPredictiveSystem:
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

    Three variants, chosen by what is passed to ``fit``:

    * standard: neither ``sigma`` nor ``groups``;
    * normalized: ``sigma`` gives a positive difficulty estimate per sample,
      so harder samples get wider distributions;
    * Mondrian: ``groups`` gives a category per sample, and each category is
      calibrated separately.

    Works on predictions, so it fits any regression model.
    """

    def __init__(self):
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

    def fit(self, y_cal, y_pred_cal, sigma_cal=None, groups_cal=None):
        """Calibrate on data not used to train the model."""
        y = np.asarray(y_cal, dtype=float).ravel()
        p = np.asarray(y_pred_cal, dtype=float).ravel()
        if y.shape != p.shape:
            raise ValueError("y_cal and y_pred_cal have different lengths.")
        if y.size == 0:
            raise ValueError("The calibration set is empty.")
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
            raise ValueError("y_cal and y_pred_cal must be finite.")
        s = self._sigma(sigma_cal, y.size, "sigma_cal")
        a = (y - p) if s is None else (y - p) / s
        self.normalized_ = s is not None
        self.mondrian_ = groups_cal is not None
        if groups_cal is None:
            self._groups = {None: _Scores(a)}
        else:
            g = np.asarray(groups_cal).ravel()
            if g.shape[0] != y.size:
                raise ValueError("groups_cal has the wrong length.")
            self._groups = {k: _Scores(a[g == k]) for k in np.unique(g).tolist()}
        return self

    # ------------------------------------------------------------- plumbing

    def _prepare(self, y_pred, sigma, groups):
        if self._groups is None:
            raise RuntimeError("Call fit() first.")
        p = np.asarray(y_pred, dtype=float).ravel()
        if self.normalized_ and sigma is None:
            raise ValueError("This system was fitted with sigma_cal; pass sigma.")
        if not self.normalized_ and sigma is not None:
            raise ValueError("This system was fitted without sigma_cal; do not pass sigma.")
        s = self._sigma(sigma, p.size, "sigma")
        if self.mondrian_:
            if groups is None:
                raise ValueError("This system was fitted with groups_cal; pass groups.")
            g = np.asarray(groups).ravel()
            if g.shape[0] != p.size:
                raise ValueError("groups has the wrong length.")
            parts = []
            for k in np.unique(g).tolist():
                if k not in self._groups:
                    raise ValueError(f"Group {k!r} was not seen in calibration.")
                parts.append((self._groups[k], np.flatnonzero(g == k)))
        else:
            if groups is not None:
                raise ValueError("This system was fitted without groups_cal; do not pass groups.")
            parts = [(self._groups[None], slice(None))]
        return p, (np.ones(p.size) if s is None else s), parts

    # ------------------------------------------------------------ quantiles

    def predict_quantile(self, y_pred, q, sigma=None, groups=None, side="higher"):
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
        p, s, parts = self._prepare(y_pred, sigma, groups)
        out = np.empty((p.size, qs.size))
        for sc, rows in parts:
            ext = np.concatenate([[-np.inf], sc.a, [np.inf]])
            a_q = ext[[min(max(_rank(sc.n, t, side), 0), sc.n + 1) for t in qs]]
            with np.errstate(invalid="ignore"):
                out[rows] = p[rows, None] + s[rows, None] * a_q[None, :]
        return out[:, 0] if scalar else out

    def predict_interval(self, y_pred, alpha=0.1, sigma=None, groups=None):
        """Central prediction intervals of shape (n_samples, 2).

        Covers with probability at least ``1 - alpha``. A bound is infinite
        when the calibration set (of that group) has fewer than
        ``2 / alpha - 1`` points.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        lo = self.predict_quantile(y_pred, alpha / 2.0, sigma, groups, side="lower")
        hi = self.predict_quantile(y_pred, 1.0 - alpha / 2.0, sigma, groups, side="higher")
        return np.column_stack([lo, hi])

    # ------------------------------------------------------------------ cdf

    def predict_cdf(self, y_pred, y, sigma=None, groups=None, tau=0.5):
        """Value ``Q(y, tau)`` of each predictive distribution at ``y``.

        ``y`` is one threshold per sample, or a single number for all.
        ``tau`` is a number or an array in [0, 1]. Pass uniform random numbers
        to get exactly uniform values at the true outcomes; the default 0.5
        is the deterministic midpoint.
        """
        p, s, parts = self._prepare(y_pred, sigma, groups)
        yv = np.broadcast_to(np.asarray(y, dtype=float), p.shape)
        tv = np.broadcast_to(np.asarray(tau, dtype=float), p.shape)
        if np.any(tv < 0) or np.any(tv > 1):
            raise ValueError("tau must be between 0 and 1.")
        z = (yv - p) / s
        out = np.empty(p.size)
        for sc, rows in parts:
            lt = np.searchsorted(sc.a, z[rows], side="left")
            le = np.searchsorted(sc.a, z[rows], side="right")
            out[rows] = (lt + tv[rows] * (le - lt + 1)) / (sc.n + 1)
        return out

    def predict_distribution(self, y_pred, sigma=None):
        """Support points ``C_1 <= ... <= C_n`` of each distribution, as an
        array of shape (n_samples, n_calibration). Not available for Mondrian
        systems, whose groups can differ in size."""
        if self._groups is not None and self.mondrian_:
            raise NotImplementedError(
                "predict_distribution is not available for Mondrian systems; "
                "use predict_quantile or predict_cdf."
            )
        p, s, parts = self._prepare(y_pred, sigma, None)
        return p[:, None] + s[:, None] * parts[0][0].a[None, :]

    # ----------------------------------------------------------------- crps

    def crps(self, y_pred, y_true, sigma=None, groups=None):
        """Continuous ranked probability score per sample (lower is better).

        Scores the distribution that puts mass ``1 / n`` on each support
        point. Computed in closed form from prefix sums, in O(log n) per
        sample, without building the distributions.
        """
        p, s, parts = self._prepare(y_pred, sigma, groups)
        yt = np.asarray(y_true, dtype=float).ravel()
        if yt.shape != p.shape:
            raise ValueError("y_pred and y_true have different lengths.")
        z = (yt - p) / s
        out = np.empty(p.size)
        for sc, rows in parts:
            zz = z[rows]
            k = np.searchsorted(sc.a, zz, side="right")
            below = sc.prefix[k]
            # sum over i of |a_i - z|
            abs_sum = zz * k - below + (sc.prefix[-1] - below) - zz * (sc.n - k)
            out[rows] = s[rows] * (abs_sum / sc.n - 0.5 * sc.gmd)
        return out
