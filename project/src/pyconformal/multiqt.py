"""MultiQT: calibrated quantile forecasts at many levels at once, online.

Wraps any quantile (or point) forecaster. After each outcome it moves a
hidden offset for every level by a quantile-tracker step and plays the
isotonic projection of ``base + offset``, so the quantiles never cross.
For any sequence with bounded forecast errors, each level's quantile is
at or above the outcome the right share of the time in the long run.

Reference
---------
Ding, Gibbs and Tibshirani (2025). Calibrated multi-level quantile
forecasting. arXiv:2512.23671 (Procedure 1; learning-rate rule of their
Section 5).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ._isotonic import isotonic_rows

__all__ = ["MultiQT", "QuantileResult"]


@dataclass
class QuantileResult:
    """Quantile forecasts from an online run.

    ``quantiles`` has shape (n_rounds, n_series, n_levels); ``levels`` are
    the probability levels; ``y`` the outcomes (n_rounds, n_series).
    """

    quantiles: np.ndarray
    levels: np.ndarray
    y: np.ndarray

    @property
    def frequencies(self) -> np.ndarray:
        """Share of outcomes at or below each level's quantile, per series: (n_series, n_levels)."""
        ok = ~np.isnan(self.y)
        hit = (self.y[:, :, None] <= self.quantiles) & ok[:, :, None]
        return hit.sum(axis=0) / np.maximum(ok.sum(axis=0), 1)[:, None]

    @property
    def calibration_error(self) -> np.ndarray:
        """Mean over levels of |frequency - level|, per series."""
        return np.abs(self.frequencies - self.levels).mean(axis=1)

    @property
    def pinball(self) -> np.ndarray:
        """Mean pinball loss over rounds and levels, per series."""
        d = self.y[:, :, None] - self.quantiles
        return np.nanmean(np.maximum(self.levels * d, (self.levels - 1) * d), axis=(0, 2))

    def interval(self, alpha: float = 0.1) -> np.ndarray:
        """Central interval between the levels alpha/2 and 1 - alpha/2: (n_rounds, n_series, 2).
        Both levels must be on the grid; otherwise a ``ValueError`` names the intervals that are available."""
        lo = int(np.argmin(np.abs(self.levels - alpha / 2)))
        hi = int(np.argmin(np.abs(self.levels - (1 - alpha / 2))))
        if abs(self.levels[lo] - alpha / 2) > 1e-9 or abs(self.levels[hi] - (1 - alpha / 2)) > 1e-9:
            ok = sorted({round(float(2 * a), 10) for a in self.levels if a < 0.5 and np.any(np.abs(self.levels - (1 - a)) < 1e-9)})
            raise ValueError(f"levels {alpha / 2:g} and {1 - alpha / 2:g} are not both on the grid; available alpha: {ok}")
        return self.quantiles[:, :, [lo, hi]]

    def cdf(self, y) -> np.ndarray:
        """Probability that the outcome is at most ``y`` (scalar or (n_rounds, n_series)), read off the quantile grid.

        A grid of quantiles does not fix the tails, so a convention is needed: linear interpolation between the
        quantiles, with the mass below the lowest level placed at the lowest quantile and the mass above the
        highest level placed at the highest quantile. The result is a proper distribution function (right-continuous,
        0 below the lowest quantile, 1 from the highest quantile on). :meth:`cdf_bounds` gives the range of
        probabilities that the grid alone supports."""
        y = np.broadcast_to(np.asarray(y, dtype=float), self.quantiles.shape[:2])
        Q, L = self.quantiles, self.levels
        K = L.size
        k = (Q <= y[:, :, None]).sum(axis=2)                              # knots at or below y
        a = np.clip(k - 1, 0, K - 1)
        b = np.clip(k, 0, K - 1)
        qa = np.take_along_axis(Q, a[:, :, None], 2)[:, :, 0]
        qb = np.take_along_axis(Q, b[:, :, None], 2)[:, :, 0]
        gap = np.where(qb > qa, qb - qa, 1.0)
        mid = L[a] + (L[b] - L[a]) * (y - qa) / gap
        return np.where(k == 0, 0.0, np.where(k == K, 1.0, mid))

    def cdf_bounds(self, y):
        """Lowest and highest probability of ``outcome <= y`` consistent with the grid: two arrays (n_rounds, n_series)."""
        y = np.broadcast_to(np.asarray(y, dtype=float), self.quantiles.shape[:2])
        k = (self.quantiles <= y[:, :, None]).sum(axis=2)
        L = np.concatenate([[0.0], self.levels, [1.0]])
        return L[k], L[k + 1]


def _row_quantile(A, p):
    """``np.nanquantile(A, p, axis=1)`` (linear method, same arithmetic), without the per-row loop; 0 for all-nan rows."""
    n, L = A.shape
    cnt = L - np.isnan(A).sum(axis=1)
    if cnt.min() == L:                                    # nothing missing: two order statistics by partition
        h = (L - 1) * p
        lo, hi = int(np.floor(h)), min(int(np.floor(h)) + 1, L - 1)
        P = np.partition(A, (lo, hi), axis=1)
        a, b, t = P[:, lo], P[:, hi], h - lo
    else:
        S = np.sort(A, axis=1)                            # nan sorts last
        h = np.maximum(cnt - 1, 0) * p
        lo = np.floor(h).astype(int)
        hi = np.minimum(lo + 1, np.maximum(cnt - 1, 0))
        r = np.arange(n)
        a, b, t = S[r, lo], S[r, hi], h - lo
    d = b - a
    out = np.where(t >= 0.5, b - d * (1 - t), a + d * t)
    return np.where(cnt > 0, out, 0.0)


class MultiQT:
    """Multi-level quantile tracker (Ding, Gibbs and Tibshirani, 2025).

    Parameters
    ----------
    levels : sequence of float
        Probability levels in (0, 1), increasing.
    lr : float or "adaptive", default="adaptive"
        Step size, in the units of the outcome. ``"adaptive"`` is the
        paper's rule: ``max(lr_floor, 0.1 * q90)``, where ``q90`` is the 90%
        quantile of the absolute errors of the base forecasts, over all
        levels, for the outcomes observed in the last ``lr_window`` rounds.
    lr_window : int, default=50
    lr_floor : float, default=0.1
        Lower limit of the adaptive step, in outcome units. Lower it for
        outcomes much smaller than 1.
    init : float or array, default=0.0
        Starting offsets.

    Notes
    -----
    Guarantee (the paper's Theorems 1 and 2), for a fixed step: if the
    errors of the base forecasts are bounded, the share of rounds with
    ``y <= q_level`` converges to the level, for every level, for any
    sequence. The adaptive step is the paper's heuristic and is not covered
    by the theorems: sequences exist on which it stays miscalibrated. The
    quantiles are ordered at every round with either step. Outcomes that arrive late or never are handled: an offset
    moves only when an outcome is observed, using the forecast that was
    issued for that round.

    Examples
    --------
    >>> mq = MultiQT(levels=[0.1, 0.5, 0.9])
    >>> res = mq.run(base=base_quantiles, y=y)        # base: (n_rounds, 3)
    >>> res.quantiles[:, 0, :]                        # adjusted, ordered
    """

    def __init__(self, levels, lr="adaptive", lr_window: int = 50, lr_floor: float = 0.1, init=0.0):
        self.levels = np.asarray(levels, dtype=float).ravel()
        if self.levels.size == 0 or np.any(self.levels <= 0) or np.any(self.levels >= 1) or np.any(np.diff(self.levels) <= 0):
            raise ValueError("levels must be increasing and inside (0, 1).")
        if not (lr == "adaptive" or (isinstance(lr, (int, float)) and lr > 0)):
            raise ValueError('lr must be a positive number or "adaptive".')
        self.lr, self.lr_window, self.lr_floor, self.init = lr, int(lr_window), float(lr_floor), init
        self._started = False

    def start(self, n_series: int = 1):
        """Reset, for ``n_series`` series tracked side by side (each on its own)."""
        K = self.levels.size
        self.n_series_ = int(n_series)
        self.offsets_ = np.zeros((self.n_series_, K)) + np.asarray(self.init, dtype=float)
        self._errors = deque(maxlen=self.lr_window)      # one (n_series, K) array per round; nan = not observed
        self._started = True
        return self

    def predict(self, base):
        """Ordered quantiles for the coming round. ``base``: (n_levels,) or (n_series, n_levels) base quantiles,
        or a point forecast per series of shape (n_series, 1)."""
        if not self._started:
            self.start(1 if np.ndim(base) < 2 else np.shape(base)[0])
        b = np.broadcast_to(np.atleast_2d(np.asarray(base, dtype=float)), self.offsets_.shape)
        return isotonic_rows(b + self.offsets_)

    def begin_round(self):
        """Open a round. Call before :meth:`update`."""
        self._errors.append(np.full((self.n_series_, self.levels.size), np.nan))   # slot for this round's own outcomes
        self._grads = []

    def update(self, y, played, base, observed=None, age: int = 0):
        """Feed outcomes. ``y`` (n_series,); ``played`` and ``base`` are the arrays of the round the outcomes
        belong to, which was ``age`` rounds ago; ``observed``: bool per series (default: all that are not nan).
        Call between :meth:`begin_round` and :meth:`end_round`."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        obs = ~np.isnan(y) if observed is None else (np.atleast_1d(observed).astype(bool) & ~np.isnan(y))
        b = np.broadcast_to(np.atleast_2d(np.asarray(base, dtype=float)), self.offsets_.shape)
        if age < len(self._errors):                       # the adaptive step looks at outcomes of the last lr_window rounds
            slot = self._errors[-1 - age]
            slot[obs] = np.abs(y[:, None] - b)[obs]
        self._grads.append((obs, (y[:, None] <= played) - self.levels[None, :]))

    def end_round(self):
        """Apply the steps for the outcomes fed since :meth:`begin_round`."""
        if not self._grads:
            return
        if self.lr == "adaptive":
            eta = np.maximum(self.lr_floor, 0.1 * _row_quantile(np.concatenate(list(self._errors), axis=1), 0.9))
        else:
            eta = np.full(self.n_series_, float(self.lr))
        for obs, g in self._grads:
            self.offsets_ -= np.where(obs[:, None], eta[:, None] * g, 0.0)

    def run(self, base, y, delay: int = 0, observed=None):
        """Replay. ``y``: (n_rounds,) or (n_rounds, n_series). ``base``: (n_rounds, n_levels) or
        (n_rounds, n_series, n_levels), or point forecasts shaped like ``y``. ``delay``: the outcome of round t
        is seen after round t + delay. ``observed``: bool like ``y``; ``False`` where the outcome is never seen."""
        Y = np.asarray(y, dtype=float)
        Y = Y[:, None] if Y.ndim == 1 else Y
        T, n = Y.shape
        K = self.levels.size
        B = np.asarray(base, dtype=float)
        if B.shape == np.asarray(y).shape:
            B = np.repeat(B.reshape(T, n, 1), K, axis=2)
        B = B.reshape(T, -1, K)
        B = np.broadcast_to(B, (T, n, K))
        O = np.ones((T, n), dtype=bool) if observed is None else np.broadcast_to(np.asarray(observed, dtype=bool).reshape(T, -1), (T, n))
        self.start(n)
        Q = np.empty((T, n, K))
        for t in range(T):
            Q[t] = self.predict(B[t])
            self.begin_round()
            s = t - int(delay)
            if s >= 0:
                self.update(Y[s], Q[s], B[s], O[s], age=int(delay))
            self.end_round()
        return QuantileResult(Q, self.levels, Y)
