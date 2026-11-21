"""Multi-step split conformal prediction for forecasts several steps ahead.

A forecaster that predicts ``H`` steps ahead has a different error
distribution at every horizon, and the error of an h-step forecast is only
known h steps later. Multi-step split conformal prediction (MSCP) keeps one
calibration set per horizon and, for every target time, uses only the errors
that were already known when that forecast was made.

Reference
---------
Wang and Hyndman (2024). Online conformal inference for multi-step time
series forecasting. arXiv:2410.13115. (R package conformalForecast.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyconformal import _acmcp_kernel as _k
from pyconformal._online_kernel import aci_replay
from pyconformal._quantile import _check_alpha, conformal_quantile

__all__ = ["MultiStepSplitConformal", "AcMCP", "MultiStepResult"]


@dataclass
class MultiStepResult:
    """``intervals[t, h - 1]`` is (lower, upper) for target time ``t`` from
    the h-step forecast; NaN where there was no forecast or too little
    calibration data. ``errors`` marks the targets that fell outside."""

    intervals: np.ndarray
    errors: np.ndarray

    @property
    def coverage(self):
        """Coverage per horizon, over the targets that had an interval."""
        have = ~np.isnan(self.intervals[:, :, 0])
        with np.errstate(invalid="ignore", divide="ignore"):
            return 1.0 - (self.errors & have).sum(axis=0) / have.sum(axis=0)


class MultiStepSplitConformal:
    """Multi-step split conformal prediction (MSCP).

    Works on forecasts and outcomes, so it fits any forecaster::

        ms = MultiStepSplitConformal(alpha=0.1)
        result = ms.run(y_pred=F, y=y)          # replay a history
        result.intervals                        # (T, H, 2)
        ms.predict_interval(y_pred=next_H)      # the H steps after the history

    ``F[t, h - 1]`` is the forecast for time ``t`` made ``h`` steps earlier
    (NaN where no such forecast exists, as in the first rows).

    Parameters
    ----------
    alpha : float, default=0.1
    symmetric : bool, default=False
        False: the lower and upper ends are calibrated separately, each at
        ``alpha / 2``, so the interval can be lopsided. True: one quantile of
        the absolute errors at ``alpha``.
    window : int or None
        Use only the most recent ``window`` errors of each horizon. None
        uses all of them.
    min_calibration : int, default=10
        No interval is issued for a horizon until this many of its errors
        are known.

    Under exchangeable errors, coverage at each horizon is at least
    ``1 - alpha``. Forecast errors are rarely exchangeable; use a window, or
    the adaptive methods, when the series drifts.
    """

    def __init__(self, alpha: float = 0.1, symmetric: bool = False, window=None, min_calibration: int = 10):
        self.alpha = _check_alpha(alpha)
        self.symmetric = bool(symmetric)
        if window is not None and int(window) < 1:
            raise ValueError("window must be a positive integer or None.")
        if int(min_calibration) < 1:
            raise ValueError("min_calibration must be at least 1.")
        self.window = None if window is None else int(window)
        self.min_calibration = int(min_calibration)
        self.errors_ = None

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in ("alpha", "symmetric", "window", "min_calibration")}

    @staticmethod
    def _inputs(y_pred, y):
        F = np.asarray(y_pred, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if F.ndim != 2 or F.shape[0] != y.shape[0]:
            raise ValueError("y_pred must have shape (T, H) with one row per outcome in y.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite.")
        return F, y

    def _levels(self, alpha):
        return (alpha, alpha) if self.symmetric else (alpha / 2.0, alpha / 2.0)

    def run(self, X=None, y=None, *, y_pred=None, alpha=None) -> MultiStepResult:
        """Intervals for every target time of a history. The interval for
        time ``t`` at horizon ``h`` uses the h-step errors of targets up to
        ``t - h`` only: what was known when the forecast was made. Afterwards
        the object holds all errors, ready for ``predict_interval``."""
        if X is not None:
            raise TypeError("Pass forecasts as y_pred= (shape (T, H)).")
        if y is None or y_pred is None:
            raise TypeError("run() needs y_pred= and y=.")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        F, y = self._inputs(y_pred, y)
        T, H = F.shape
        E = y[:, None] - F  # error of the h-step forecast for each target
        out = np.full((T, H, 2), np.nan)
        a_lo, a_hi = self._levels(alpha)
        w = self.window or 0
        for h in range(1, H + 1):
            rows = np.flatnonzero(~np.isnan(E[:, h - 1]))
            e = E[rows, h - 1]
            n = e.shape[0]
            if n == 0:
                continue
            # q[j] = conformal quantile of the first j errors (in a window)
            if self.symmetric:
                q_up = aci_replay(np.empty(0), np.abs(e), a_hi, 0.0, a_hi, w)[0]
                q_lo = q_up
            else:
                q_up = aci_replay(np.empty(0), e, a_hi, 0.0, a_hi, w)[0]
                q_lo = aci_replay(np.empty(0), -e, a_lo, 0.0, a_lo, w)[0]
            # target rows[j] may use the errors of targets up to rows[j] - h
            known = np.searchsorted(rows, rows - h, side="right")
            ok = known >= self.min_calibration
            # with `known` errors available, the replay threshold to read is the
            # one computed just before the error number `known` arrived; when
            # known == n it was never computed, so compute it directly
            idx = np.minimum(known, n - 1)
            lo = np.where(ok, q_lo[idx], np.nan)
            up = np.where(ok, q_up[idx], np.nan)
            last = ok & (known == n)
            if np.any(last):  # cannot happen for h >= 1, kept for safety
                lo[last], up[last] = self._tail(e, a_lo, a_hi)
            out[rows, h - 1, 0] = F[rows, h - 1] - lo
            out[rows, h - 1, 1] = F[rows, h - 1] + up
        self.errors_ = E
        with np.errstate(invalid="ignore"):
            miss = (y[:, None] < out[:, :, 0]) | (y[:, None] > out[:, :, 1])
        return MultiStepResult(out, miss)

    def _tail(self, e, a_lo, a_hi):
        if self.window is not None:
            e = e[-self.window :]
        if self.symmetric:
            q = conformal_quantile(np.abs(e), a_hi)
            return q, q
        return conformal_quantile(-e, a_lo), conformal_quantile(e, a_hi)

    def calibrate(self, X=None, y=None, *, y_pred=None):
        """Store the errors of a history without computing its intervals."""
        if X is not None:
            raise TypeError("Pass forecasts as y_pred= (shape (T, H)).")
        if y is None or y_pred is None:
            raise TypeError("calibrate() needs y_pred= and y=.")
        F, y = self._inputs(y_pred, y)
        self.errors_ = y[:, None] - F
        return self

    def predict_interval(self, X=None, alpha=None, *, y_pred=None):
        """Intervals for the next ``H`` steps after the history, shape
        (H, 2): row ``h - 1`` is the h-step forecast ``y_pred[h - 1]``,
        calibrated on every h-step error seen so far."""
        if self.errors_ is None:
            raise RuntimeError("Call run() or calibrate() first.")
        if X is not None or y_pred is None:
            raise TypeError("Pass the next forecasts as y_pred= (length H).")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        f = np.asarray(y_pred, dtype=float).ravel()
        H = self.errors_.shape[1]
        if f.shape[0] != H:
            raise ValueError(f"Expected {H} forecasts, one per horizon.")
        a_lo, a_hi = self._levels(alpha)
        out = np.full((H, 2), np.nan)
        for h in range(H):
            e = self.errors_[:, h]
            e = e[~np.isnan(e)]
            if e.shape[0] >= self.min_calibration:
                lo, up = self._tail(e, a_lo, a_hi)
                out[h] = f[h] - lo, f[h] + up
        return out


class AcMCP:
    """Autocorrelated multi-step conformal prediction (Wang and Hyndman).

    An online method: for each horizon the half-widths of the interval are
    tracked from past misses, with three parts added together.

    * Quantile tracking: widen after a miss, narrow after a hit, with a step
      proportional to the range of the errors seen.
    * An integrator that pushes harder the longer coverage stays off target.
    * A scorecaster that forecasts the next error. It uses the fact that the
      errors of an optimal h-step forecast are at most MA(h - 1): the
      forecast is the average of the mean of an MA(h - 1) model and a
      regression of the h-step error on the shorter horizons.

    Use it like :class:`MultiStepSplitConformal`::

        ac = AcMCP(alpha=0.1, min_calibration=50)
        result = ac.run(y_pred=F, y=y)
        ac.predict_interval(y_pred=next_H)

    Forecasts may be missing (NaN) only in the first rows of each column.

    Parameters
    ----------
    alpha : float, default=0.1
        Each end of the interval is tracked at ``alpha / 2``.
    min_calibration : int, default=10
        Intervals and scorecasts start once a horizon has this many errors.
    window : int or None
        Learning rate, MA model and regression use only the most recent
        ``window`` errors. None uses all of them. ``window ==
        min_calibration`` is ``rolling = TRUE`` in the R package.
    integrate, scorecast : bool, default=True
    lr : float, default=0.1
    KI : float or None
        Scale of the integrator. None uses the largest absolute error seen
        so far at that horizon. (The R package's default is the largest
        absolute error of the whole history, which includes errors that are
        not yet known at the time; pass that number to reproduce it.)
    Csat : float or None
        Integrator saturation. None uses the formula of the conformal PID
        paper with the length of the history and ``delta``.
    delta : float, default=0.01
    refit_every : int, default=1
        Refit the MA model every this many steps. 1 follows the R package.

    The long-run coverage of each end is guaranteed by quantile tracking and
    the integrator whatever the data; the scorecaster only makes intervals
    tighter when the errors are predictable.

    Reference
    ---------
    Wang and Hyndman (2024), arXiv:2410.13115.
    """

    _params = ("alpha", "min_calibration", "window", "integrate", "scorecast", "lr", "KI", "Csat", "delta", "refit_every")

    def __init__(self, alpha: float = 0.1, min_calibration: int = 10, window=None, integrate: bool = True,
                 scorecast: bool = True, lr: float = 0.1, KI=None, Csat=None, delta: float = 0.01,
                 refit_every: int = 1):
        self.alpha = _check_alpha(alpha)
        if int(min_calibration) < 2:
            raise ValueError("min_calibration must be at least 2.")
        if window is not None and int(window) < 2:
            raise ValueError("window must be at least 2, or None.")
        if not lr > 0 or (KI is not None and KI < 0) or (Csat is not None and not Csat > 0):
            raise ValueError("lr and Csat must be positive and KI non-negative.")
        if not 0 < delta < 1 or int(refit_every) < 1:
            raise ValueError("delta must be in (0, 1) and refit_every at least 1.")
        self.min_calibration = int(min_calibration)
        self.window = None if window is None else int(window)
        self.integrate, self.scorecast = bool(integrate), bool(scorecast)
        self.lr = float(lr)
        self.KI = None if KI is None else float(KI)
        self.Csat = None if Csat is None else float(Csat)
        self.delta, self.refit_every = float(delta), int(refit_every)
        self.next_ = None

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in self._params}

    def run(self, X=None, y=None, *, y_pred=None, alpha=None) -> MultiStepResult:
        """Intervals for every target time of a history; the interval for
        time ``t`` at horizon ``h`` uses only what was known at ``t - h``."""
        if X is not None:
            raise TypeError("Pass forecasts as y_pred= (shape (T, H)).")
        if y is None or y_pred is None:
            raise TypeError("run() needs y_pred= and y=.")
        alpha = self.alpha if alpha is None else _check_alpha(alpha)
        F, y = MultiStepSplitConformal._inputs(y_pred, y)
        T, H = F.shape
        E = np.ascontiguousarray(y[:, None] - F)
        ok = ~np.isnan(E)
        starts = np.where(ok.any(axis=0), ok.argmax(axis=0), T).astype(np.int64)
        for h in range(H):
            if not ok[starts[h] :, h].all():
                raise ValueError("Forecasts may be missing only in the first rows of each column.")
        if self.Csat is None:
            if T < 3:
                raise ValueError("The history is too short to set Csat; pass Csat.")
            csat = 2.0 / np.pi * (np.ceil(np.log(T) * self.delta) - 1.0 / np.log(T))
            if not csat > 0:
                raise ValueError("The default Csat is not positive for this history; pass Csat.")
        else:
            csat = self.Csat
        ncal, w = self.min_calibration, self.window or 0
        SC = np.full((T + H, H), np.nan)
        if self.scorecast:
            MA = np.full((T, H), np.nan)
            if H > 1:
                _k.ma_means(E, starts, ncal, w, self.refit_every, MA)
            _k.scorecast(E, starts, ncal, w, MA, SC)
        out = np.full((T, H, 2), np.nan)
        nxt = np.full((H, 2), np.nan)
        ki = -1.0 if self.KI is None else self.KI
        for h in range(1, H + 1):
            s = int(starts[h - 1])
            if s >= T:
                continue
            e = np.ascontiguousarray(E[:, h - 1])
            sc = np.ascontiguousarray(SC[:, h - 1])
            first = s + ncal - 1 + h  # first target with an interval
            for side, sign in ((0, -1.0), (1, 1.0)):
                qs = np.zeros(T + H)
                _k.track(e, s, h, ncal, w, alpha / 2.0, self.lr, ki, csat, self.integrate, sc, sign, qs)
                if first < T:
                    out[first:, h - 1, side] = F[first:, h - 1] + sign * qs[first:T]
                if T - 1 + h >= first:
                    nxt[h - 1, side] = qs[T - 1 + h]
        self.next_ = nxt
        self.scorecasts_ = SC[:T] if self.scorecast else None
        with np.errstate(invalid="ignore"):
            miss = (y[:, None] < out[:, :, 0]) | (y[:, None] > out[:, :, 1])
        return MultiStepResult(out, miss)

    def predict_interval(self, X=None, *, y_pred=None):
        """Intervals for the next ``H`` steps after the history, shape (H, 2),
        at the level used in ``run``."""
        if self.next_ is None:
            raise RuntimeError("Call run() first.")
        if X is not None or y_pred is None:
            raise TypeError("Pass the next forecasts as y_pred= (length H).")
        f = np.asarray(y_pred, dtype=float).ravel()
        if f.shape[0] != self.next_.shape[0]:
            raise ValueError(f"Expected {self.next_.shape[0]} forecasts, one per horizon.")
        return np.column_stack([f - self.next_[:, 0], f + self.next_[:, 1]])
