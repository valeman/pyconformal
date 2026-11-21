"""Conformal Seasonal Pools (CSP): a training-free probabilistic forecaster.

Reference
---------
Manokhin, V. Training-Free Probabilistic Time-Series Forecasting with
Conformal Seasonal Pools. COPA 2026.

This is a vectorised implementation of the method in ``csp-forecaster``
0.2.0 (``mode="fast"``). It builds the same predictive distribution: the
same seasonal pools and recency weights, the same point forecasts and the
same residual pools. It draws from that distribution without a loop over
horizons:

* the recency weights of a seasonal pool are geometric, so the weighted
  draw has a closed-form inverse CDF and needs no cumulative-weight table;
* a residual at lag ``L`` is ``y[i + L] - y[i]``, so residual pools are
  never materialised: a draw is two indexed reads of the history.

Draws use a different stream of random numbers than ``csp-forecaster``, so
samples are equal in distribution, not value by value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConformalSeasonalPool", "CSPResult", "DEFAULT_QUANTILE_LEVELS"]

DEFAULT_QUANTILE_LEVELS = (0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995)


@dataclass
class CSPResult:
    """Predictive distribution for an H-step forecast.

    Attributes
    ----------
    lower, upper : ndarray of shape (H,)
        Bounds of the central ``1 - alpha`` interval.
    quantiles : dict
        Maps each requested level to an array of shape (H,).
    samples : ndarray of shape (H, n_samples)
        Predictive draws per horizon.
    """

    lower: np.ndarray
    upper: np.ndarray
    quantiles: dict
    samples: np.ndarray
    alpha: float
    metadata: dict = field(default_factory=dict)


class ConformalSeasonalPool:
    """Conformal Seasonal Pool forecaster.

    For each horizon, predictive samples are a blend of

    1. a seasonal pool: past values at the same seasonal phase, drawn with
       exponential recency weights, and
    2. a conformal component: a seasonal-naive point forecast plus a signed
       residual drawn from past seasonal-naive errors.

    No training and no learned parameters. This is a forecasting heuristic
    built from conformal residuals; it does not carry a finite-sample
    coverage guarantee for dependent data.

    Parameters
    ----------
    pool_weight : float, default=0.5
        Share of samples drawn from the seasonal pool.
    exp_lambda : float, default=0.01
        Recency-decay rate of the seasonal pool (0 for uniform weights).
    cal_fraction : float, default=0.5
        Share of the history used as the residual calibration window.
    adaptive : bool, default=True
        Drop the seasonal pool when there is no seasonality, and cap its
        weight at 0.3 when fewer than three full cycles are available.
    residual_mode : {"h_step", "paper"}, default="h_step"
        ``"h_step"`` uses residuals at the seasonal-naive multi-step lag
        ``m * ceil(h / m)`` for horizon ``h``, so intervals widen with the
        horizon. ``"paper"`` uses one residual pool for every horizon.
    orientation : bool, default=False
        Push interval quantiles outward by the finite-sample correction
        (``floor((n + 1) q) / n`` below the median, ``ceil((n + 1) q) / n``
        above it).
    decay_unit : {"step", "cycle"}, default="step"
        Whether recency decay counts time steps or seasonal cycles.
    random_state : int, numpy.random.Generator or None
        Seed or generator for the draws.
    """

    MIN_RESIDUALS = 8

    def __init__(
        self,
        pool_weight: float = 0.5,
        exp_lambda: float = 0.01,
        cal_fraction: float = 0.5,
        adaptive: bool = True,
        residual_mode: str = "h_step",
        orientation: bool = False,
        decay_unit: str = "step",
        random_state=None,
    ):
        if residual_mode not in ("paper", "h_step"):
            raise ValueError(f"residual_mode must be 'paper' or 'h_step', got {residual_mode!r}.")
        if decay_unit not in ("cycle", "step"):
            raise ValueError(f"decay_unit must be 'cycle' or 'step', got {decay_unit!r}.")
        if not 0.0 <= pool_weight <= 1.0:
            raise ValueError("pool_weight must be between 0 and 1.")
        if not exp_lambda >= 0.0:
            raise ValueError("exp_lambda must be >= 0.")
        if not 0.0 < cal_fraction <= 1.0:
            raise ValueError("cal_fraction must be in (0, 1].")
        self.pool_weight = float(pool_weight)
        self.exp_lambda = float(exp_lambda)
        self.cal_fraction = float(cal_fraction)
        self.adaptive = bool(adaptive)
        self.residual_mode = residual_mode
        self.orientation = bool(orientation)
        self.decay_unit = decay_unit
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self.history = None
        self.seasonal_period = 1

    # ------------------------------------------------------------------ fit

    def fit(self, history, seasonal_period: int = 1):
        """Store the series (1-D, most recent value last) and its season length."""
        y = np.asarray(history, dtype=np.float64).ravel()
        if y.size == 0:
            raise ValueError("history is empty.")
        if not np.all(np.isfinite(y)):
            raise ValueError("history must be finite.")
        self.history = y
        self.seasonal_period = max(1, int(seasonal_period))
        return self

    # ----------------------------------------------------------------- plan

    def _effective_pool_weight(self) -> float:
        if not self.adaptive:
            return self.pool_weight
        m = self.seasonal_period
        if m <= 1:
            return 0.0
        if len(self.history) / m < 3:
            return min(self.pool_weight, 0.3)
        return self.pool_weight

    def _n_cal(self) -> int:
        T = len(self.history)
        return min(max(int(T * self.cal_fraction), self.seasonal_period + 1), T)

    def _lag_spec(self, lag: int):
        """(start, lag, size, scale) of the residual pool at ``lag``:
        residual i is ``scale * (y[start + lag + i] - y[start + i])``.
        ``size == 0`` stands for the single residual 0."""
        T = len(self.history)
        n_cal = self._n_cal()
        lag = max(1, int(lag))
        if n_cal - lag >= self.MIN_RESIDUALS:  # calibration window
            return T - n_cal, lag, n_cal - lag, 1.0
        if T - lag >= self.MIN_RESIDUALS:  # whole history, same lag
            return 0, lag, T - lag, 1.0
        shorter = min(lag - 1, T - self.MIN_RESIDUALS)
        if shorter >= 1:  # shorter lag, rescaled to the requested one
            return 0, shorter, T - shorter, math.sqrt(lag / shorter)
        if T >= 2:
            return 0, 1, T - 1, math.sqrt(lag)
        return 0, 1, 0, 1.0

    def _paper_spec(self):
        T = len(self.history)
        m = self.seasonal_period
        n_cal = self._n_cal()
        lag = m if (m > 1 and n_cal > m) else 1
        return T - n_cal, lag, max(n_cal - lag, 0), 1.0

    def _plan(self, H: int):
        """Per-horizon description of the predictive distribution."""
        T = len(self.history)
        m = self.seasonal_period
        h = np.arange(H)

        # Seasonal-naive point forecast.
        mu_idx = T - m + (h % m)
        mu_idx = np.where(mu_idx < 0, T - 1, mu_idx)
        mu = self.history[mu_idx]

        # Seasonal pool of horizon h: y[last - m * j], j = 0 .. size - 1.
        pos = (T + h) % m
        pool_size = np.where(pos < T, (T - 1 - pos) // m + 1, 0)
        pool_last = pos + m * (pool_size - 1)

        # Residual pools.
        if self.residual_mode == "h_step":
            lags = m * ((h + m) // m)  # m * ceil((h + 1) / m)
            cache = {int(L): self._lag_spec(int(L)) for L in np.unique(lags)}
            spec = np.array([cache[int(L)] for L in lags], dtype=np.float64)
        else:
            spec = np.tile(np.array(self._paper_spec(), dtype=np.float64), (H, 1))
        return {
            "mu": mu,
            "pool_size": pool_size.astype(np.int64),
            "pool_last": pool_last.astype(np.int64),
            "res_start": spec[:, 0].astype(np.int64),
            "res_lag": spec[:, 1].astype(np.int64),
            "res_size": spec[:, 2].astype(np.int64),
            "res_scale": spec[:, 3],
        }

    def _log_rho(self) -> float:
        """Log of the weight ratio between consecutive cycles of a pool."""
        unit = self.seasonal_period if self.decay_unit == "step" else 1
        return -self.exp_lambda * unit

    # -------------------------------------------------------------- predict

    def predict(self, H: int, alpha: float = 0.05, quantile_levels=None, n_samples: int = 100):
        """Forecast ``H`` steps ahead.

        Returns a :class:`CSPResult` with the central ``1 - alpha`` interval,
        the requested quantiles and the predictive samples.
        """
        if self.history is None:
            raise RuntimeError("Call fit() before predict().")
        H = int(H)
        n = int(n_samples)
        if H < 1 or n < 1:
            raise ValueError("H and n_samples must be at least 1.")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1.")
        levels = list(DEFAULT_QUANTILE_LEVELS if quantile_levels is None else quantile_levels)

        y = self.history
        m = self.seasonal_period
        pw = self._effective_pool_weight()
        n_pool = int(n * pw) if m > 1 else 0
        plan = self._plan(H)
        U = self._rng.random((H, n))
        samples = np.empty((H, n))

        def conformal(u, rows):
            size = plan["res_size"][rows][:, None]
            i = np.minimum((u * size).astype(np.int64), np.maximum(size - 1, 0))
            a = plan["res_start"][rows][:, None] + i
            hi = np.minimum(a + plan["res_lag"][rows][:, None], len(y) - 1)
            r = (y[hi] - y[a]) * plan["res_scale"][rows][:, None]
            return plan["mu"][rows][:, None] + np.where(size > 0, r, 0.0)

        every = np.arange(H)
        if n_pool < n:
            samples[:, n_pool:] = conformal(U[:, n_pool:], every)
        if n_pool > 0:
            size = plan["pool_size"][:, None]
            log_rho = self._log_rho()
            u = U[:, :n_pool]
            if log_rho < 0.0:
                # Inverse CDF of weights proportional to rho ** j.
                j = np.log1p(u * np.expm1(size * log_rho)) / log_rho
                j = j.astype(np.int64)
            else:
                j = (u * size).astype(np.int64)
            j = np.clip(j, 0, np.maximum(size - 1, 0))
            idx = np.maximum(plan["pool_last"][:, None] - m * j, 0)
            samples[:, :n_pool] = y[idx]
            # A pool with fewer than two values is not used.
            thin = np.flatnonzero(plan["pool_size"] < 2)
            if thin.size:
                samples[thin, :n_pool] = conformal(u[thin], thin)

        taus = sorted(set([alpha / 2.0, 1.0 - alpha / 2.0, *levels]))
        if self.orientation:
            ev = [
                max(0.0, math.floor((n + 1.0) * t) / n) if t < 0.5
                else min(1.0, math.ceil((n + 1.0) * t) / n)
                for t in taus
            ]
        else:
            ev = taus
        # Linear-interpolation quantiles of every row from one sort.
        srt = np.sort(samples, axis=1)
        p = np.clip(np.asarray(ev, dtype=np.float64), 0.0, 1.0) * (n - 1)
        lo = np.floor(p).astype(np.int64)
        hi = np.minimum(lo + 1, n - 1)
        frac = p - lo
        q = srt[:, lo] + (srt[:, hi] - srt[:, lo]) * frac  # (H, n_taus)
        row = {t: q[:, k] for k, t in enumerate(taus)}
        return CSPResult(
            lower=row[alpha / 2.0],
            upper=row[1.0 - alpha / 2.0],
            quantiles={t: row[t] for t in levels},
            samples=samples,
            alpha=alpha,
            metadata={
                "residual_mode": self.residual_mode,
                "orientation": self.orientation,
                "decay_unit": self.decay_unit,
                "pool_weight": self.pool_weight,
                "pool_weight_eff": pw,
                "exp_lambda": self.exp_lambda,
                "cal_fraction": self.cal_fraction,
            },
        )
