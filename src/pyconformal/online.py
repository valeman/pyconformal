"""Online conformal prediction for time series and other streams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pyconformal._online_kernel import _RANK_TOL, aci_replay
from pyconformal._quantile import _check_alpha

__all__ = ["AdaptiveConformalInference", "ACIResult"]


@dataclass
class ACIResult:
    """Output of replaying a stream.

    Attributes
    ----------
    thresholds : ndarray of shape (T,)
        Score threshold used at each step, fixed before that step's outcome
        was seen. ``inf`` means "cover everything"; ``-inf`` means the empty
        set.
    errors : ndarray of bool, shape (T,)
        True where the observed score exceeded the threshold (a miss).
    alphas : ndarray of shape (T,)
        Working miscoverage level used at each step.
    intervals : ndarray of shape (T, 2) or None
        Lower and upper bounds, when predictions were supplied.
    """

    thresholds: np.ndarray
    errors: np.ndarray
    alphas: np.ndarray
    intervals: np.ndarray | None = None

    @property
    def coverage(self) -> float:
        return float(1.0 - self.errors.mean())


class AdaptiveConformalInference:
    """Adaptive conformal inference (Gibbs and Candes, 2021).

    Split conformal prediction with a miscoverage level that reacts to past
    errors. At each step the threshold is the conformal quantile of the
    current calibration scores at the working level ``alpha_t``; after the
    outcome is seen,

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),

    where ``err_t`` is 1 for a miss and 0 otherwise, and the new score joins
    the calibration set.

    Guarantee (no assumption on the data, not even exchangeability): over
    any ``T`` steps the average miss rate differs from ``alpha`` by at most
    ``(max(alpha_1, 1 - alpha_1) + gamma) / (gamma * T)``. This holds because
    the working level is not clipped: when ``alpha_t <= 0`` the interval is
    infinite, and when ``alpha_t >= 1`` the prediction set is empty.

    The method works on nonconformity scores, so it fits any forecaster.
    The model is never called inside the update loop.

    Parameters
    ----------
    alpha : float, default=0.1
        Target miscoverage level.
    gamma : float, default=0.005
        Step size. 0 gives plain online split conformal. Larger values
        adapt faster but make the intervals less stable.
    window : int or None, default=None
        Keep only the most recent ``window`` scores for calibration.
        None keeps all scores.

    Attributes
    ----------
    alpha_t_ : float
        Current working level.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.005, window: int | None = None):
        self.alpha = _check_alpha(alpha)
        gamma = float(gamma)
        if not gamma >= 0.0 or math.isinf(gamma):
            raise ValueError(f"gamma must be a finite number >= 0, got {gamma}.")
        if window is not None:
            if int(window) != window or window < 1:
                raise ValueError(f"window must be a positive integer or None, got {window}.")
            window = int(window)
        self.gamma = gamma
        self.window = window
        self.alpha_t_ = self.alpha
        self._scores = np.empty(0)

    # ------------------------------------------------------------- scores

    @staticmethod
    def _as_scores(scores, name="scores"):
        s = np.asarray(scores, dtype=float).ravel()
        if not np.all(np.isfinite(s)):
            raise ValueError(f"{name} must be finite.")
        return s

    def _trim(self):
        if self.window is not None and self._scores.shape[0] > self.window:
            self._scores = self._scores[-self.window :]

    def fit(self, scores_cal=()):
        """Start from an initial set of calibration scores (may be empty).

        Resets the working level to ``alpha``.
        """
        self._scores = self._as_scores(scores_cal, "scores_cal").copy()
        self._trim()
        self.alpha_t_ = self.alpha
        return self

    @property
    def calibration_scores_(self):
        return self._scores.copy()

    def threshold(self) -> float:
        """Score threshold for the next step."""
        n = self._scores.shape[0]
        if self.alpha_t_ <= 0.0:
            return math.inf
        k = math.ceil((n + 1) * (1.0 - self.alpha_t_) - _RANK_TOL)
        if k > n:
            return math.inf
        if k < 1:
            return -math.inf
        return float(np.partition(self._scores, k - 1)[k - 1])

    def update(self, score) -> bool:
        """Observe the score of the step just predicted. Returns True on a miss."""
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("score must be finite.")
        miss = score > self.threshold()
        self.alpha_t_ = self.alpha_t_ + self.gamma * (self.alpha - (1.0 if miss else 0.0))
        self._scores = np.append(self._scores, score)
        self._trim()
        return miss

    def run(self, scores, *, use_numba=None) -> ACIResult:
        """Replay a whole stream of scores in one call.

        Equivalent to calling ``threshold()`` then ``update(score)`` for each
        score in turn, and leaves the object in the same state, but runs in
        O(log n) per step.
        """
        s = self._as_scores(scores)
        q, err, alphas, final = aci_replay(
            self._scores, s, self.alpha, self.gamma, self.alpha_t_, self.window or 0, use_numba
        )
        self._scores = np.concatenate([self._scores, s])
        self._trim()
        self.alpha_t_ = final
        return ACIResult(q, err, alphas)

    # ---------------------------------------------- regression convenience

    def predict_interval(self, y_pred):
        """Interval(s) for the next step around point prediction(s).

        Uses the absolute-residual score: ``y_pred -/+ threshold``. An empty
        prediction set (threshold ``-inf``) is returned as the single point
        ``y_pred``.
        """
        y_pred = np.asarray(y_pred, dtype=float)
        h = max(self.threshold(), 0.0)
        return np.stack([y_pred - h, y_pred + h], axis=-1)

    def run_intervals(self, y_pred, y_true, *, use_numba=None) -> ACIResult:
        """Replay a stream of point predictions and outcomes.

        Scores are absolute residuals ``|y_true - y_pred|``. The interval at
        step ``t`` uses only outcomes before ``t``.
        """
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true have different lengths.")
        res = self.run(np.abs(y_true - y_pred), use_numba=use_numba)
        h = np.maximum(res.thresholds, 0.0)
        res.intervals = np.column_stack([y_pred - h, y_pred + h])
        return res
