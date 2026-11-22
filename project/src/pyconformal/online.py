"""Online conformal prediction for time series and other streams."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from pyconformal._base import _Conformal

from pyconformal._online_kernel import (
    _decimal,
    _rank,
    _agaci_combine,
    _agaci_learn,
    _dtaci_step,
    aci_replay,
    agaci_replay,
    dtaci_replay,
    nexcp_replay,
    pid_replay,
)
from pyconformal._quantile import _check_alpha

__all__ = ["AdaptiveConformalInference", "ACIResult"]


@dataclass
class ACIResult:
    """Output of replaying a stream.

    Attributes
    ----------
    thresholds : ndarray of shape (T,)
        Score threshold used at each step, fixed before that step's outcome
        was seen. ``inf`` means "cover everything"; a negative threshold
        (including ``-inf``) means the empty set.
    errors : ndarray of bool, shape (T,)
        True where the observed score exceeded the threshold (a miss).
    alphas : ndarray of shape (T,)
        Working miscoverage level used at each step.
    intervals : ndarray of shape (T, 2) or None
        Lower and upper bounds, when predictions were supplied. An empty set
        is the row ``(nan, nan)``, which contains no outcome. With the
        absolute-error score an outcome is inside the issued interval
        exactly when ``errors`` is False, so ``coverage`` is also the share
        of outcomes inside the issued intervals.
    """

    thresholds: np.ndarray
    errors: np.ndarray
    alphas: np.ndarray
    intervals: np.ndarray | None = None

    @property
    def coverage(self) -> float:
        return float(1.0 - self.errors.mean())


def _band(p, q):
    """``p -/+ q`` for the absolute-error score; a negative threshold is the
    empty set, returned as NaN bounds."""
    q = np.asarray(q, dtype=float)
    h = np.where(q < 0.0, np.nan, q)
    return np.column_stack([p - h, p + h])


class AdaptiveConformalInference(_Conformal):
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

    Works on nonconformity scores (``scores=``), on predictions and
    outcomes (``y_pred=``, ``y=``; the score is the absolute error), or on a
    model (``X``, ``y``). The model is called once per batch, never inside
    the update loop.

    Parameters
    ----------
    estimator : object or None
        A model with ``predict``. None when working from scores or
        predictions.
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

    def __init__(
        self, estimator=None, alpha: float = 0.1, gamma: float = 0.005, window: int | None = None
    ):
        self.estimator = estimator
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
        self._sorted = None

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

    def _stream(self, X, y, y_pred, scores):
        """(scores, predictions or None) from whichever inputs were given."""
        if scores is not None:
            if X is not None or y is not None or y_pred is not None:
                raise TypeError("Pass scores= alone, or predictions with y.")
            return self._as_scores(scores), None
        if y is None:
            raise TypeError("Pass scores=, or the outcomes y with X or y_pred=.")
        p = self._output(X, y_pred).ravel()
        yv = np.asarray(y, dtype=float).ravel()
        if p.shape != yv.shape:
            raise ValueError("Predictions and y have different lengths.")
        return self._as_scores(np.abs(yv - p)), p

    def _forget_calibration(self):
        self.calibrate()

    def calibrate(self, X=None, y=None, *, y_pred=None, scores=None):
        """Start from an initial calibration set. Calling it with no
        arguments starts from nothing.

        Resets the working level to ``alpha``.
        """
        if X is None and y is None and y_pred is None and scores is None:
            scores = ()
        self._scores = self._stream(X, y, y_pred, scores)[0].copy()
        self._sorted = None
        self._trim()
        self._reset()
        return self

    @property
    def calibration_scores_(self):
        self._sync()
        return self._scores.copy()

    def _online(self):
        """Sorted copy and arrival-ordered list of the calibration scores,
        kept only while stepping one observation at a time."""
        if self._sorted is None:
            self._arrival = self._scores.tolist()
            self._sorted = sorted(self._arrival)
            self._head = 0
        return self._sorted

    def _sync(self):
        if self._sorted is not None:
            self._scores = np.asarray(self._arrival[self._head :], dtype=float)
            self._sorted = None

    @property
    def _decimal_alpha(self):
        """(numerator, denominator) of the decimal value of ``alpha``."""
        cache = getattr(self, "_dec_cache", None)
        if cache is None or cache[0] != self.alpha:
            cache = (self.alpha, *_decimal(self.alpha))
            self._dec_cache = cache
        return cache[1], cache[2]

    def threshold(self) -> float:
        """Score threshold for the next step."""
        srt = self._online()
        n = len(srt)
        if self.alpha_t_ <= 0.0:
            return math.inf
        k = _rank(n, self.alpha_t_, self.alpha, *self._decimal_alpha)
        if k > n:
            return math.inf
        if k < 1:
            return -math.inf
        return srt[k - 1]

    def update(self, X=None, y=None, *, y_pred=None, score=None) -> bool:
        """Observe the step just predicted: its score, or its prediction and
        outcome. Returns True on a miss."""
        if score is None:
            score = self._stream(X, y, y_pred, None)[0]
            if score.shape[0] != 1:
                raise ValueError("update() takes one step; use run() for a stream.")
            score = score[0]
        elif X is not None or y is not None or y_pred is not None:
            raise TypeError("Pass score= alone, or a prediction with y.")
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("score must be finite.")
        miss = self._adapt(score)
        srt = self._sorted
        bisect.insort(srt, score)
        self._arrival.append(score)
        if self.window is not None and len(srt) > self.window:
            old = self._arrival[self._head]
            del srt[bisect.bisect_left(srt, old)]
            self._head += 1
            if self._head > 4096 and self._head * 2 > len(self._arrival):
                del self._arrival[: self._head]
                self._head = 0
        return miss

    def _adapt(self, score) -> bool:
        """Compare the score with the current threshold and move the working
        level. Called before the score joins the calibration set."""
        miss = score > self.threshold()
        self.alpha_t_ = self.alpha_t_ + self.gamma * (self.alpha - (1.0 if miss else 0.0))
        return miss

    def _replay(self, s, use_numba):
        q, err, alphas, final = aci_replay(
            self._scores, s, self.alpha, self.gamma, self.alpha_t_, self.window or 0, use_numba
        )
        self.alpha_t_ = final
        return q, err, alphas

    def _reset(self):
        self.alpha_t_ = self.alpha

    def run(self, X=None, y=None, *, y_pred=None, scores=None, use_numba=None) -> ACIResult:
        """Replay a whole stream in one call.

        Equivalent to calling ``threshold()`` then ``update(...)`` for each
        step in turn, and leaves the object in the same state, but runs in
        O(log n) per step. With predictions, the result also carries the
        intervals ``y_pred -/+ threshold``; the interval at step ``t`` uses
        only outcomes before ``t``.
        """
        s, p = self._stream(X, y, y_pred, scores)
        self._sync()
        q, err, alphas = self._replay(s, use_numba)
        self._scores = np.concatenate([self._scores, s])
        self._trim()
        res = ACIResult(q, err, alphas)
        if p is not None:
            res.intervals = _band(p, q)
        return res

    # ---------------------------------------------- regression convenience

    def predict_interval(self, X=None, *, y_pred=None):
        """Interval(s) for the next step, shape (n, 2).

        Uses the absolute-error score: ``y_pred -/+ threshold``. When the
        threshold is negative the prediction set is empty and the row is
        ``(nan, nan)``.
        """
        p = self._output(X, y_pred).ravel()
        return _band(p, np.full(p.shape[0], self.threshold()))


class DtACI(AdaptiveConformalInference):
    """Dynamically-tuned adaptive conformal inference (Gibbs and Candes, 2022).

    ACI needs a step size ``gamma``, and the right value depends on how fast
    the data drift. DtACI runs ACI with several step sizes at once and
    combines their working levels with weights that follow which step size
    has been doing best lately. There is no step size to choose.

    Same calls as :class:`AdaptiveConformalInference`. With a single step
    size it gives exactly the same results as ACI.

    Parameters
    ----------
    estimator : object or None
    alpha : float, default=0.1
    gammas : sequence of float or None
        Candidate step sizes. Default: 0.001, 0.002, 0.004, ..., 0.128.
    interval : int, default=500
        Length of the stretch over which the method should adapt; sets how
        fast the weights move (the paper's ``I``).
    window : int or None
        Keep only the most recent scores.

    Reference
    ---------
    Gibbs and Candes (2022). Conformal inference for online prediction with
    arbitrary distribution shifts. arXiv:2208.08401.
    """

    def __init__(self, estimator=None, alpha: float = 0.1, gammas=None, interval: int = 500, window: int | None = None):
        super().__init__(estimator, alpha, 0.0, window)
        g = 0.001 * 2.0 ** np.arange(8) if gammas is None else np.asarray(gammas, dtype=float).ravel()
        if g.size == 0 or not np.all(np.isfinite(g)) or np.any(g < 0):
            raise ValueError("gammas must be finite numbers >= 0.")
        if int(interval) < 1:
            raise ValueError("interval must be at least 1.")
        self.gammas = g
        self.interval = int(interval)
        self._reset()

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in ("estimator", "alpha", "gammas", "interval", "window")}

    @property
    def _rates(self):
        a, I, k = self.alpha, self.interval, self.gammas.size
        denom = ((1 - a) ** 2 * a**3 + a**2 * (1 - a) ** 3) / 3.0
        eta = math.sqrt(3.0 / I) * math.sqrt((math.log(I * k) + 2.0) / denom)
        return eta, 1.0 / (2.0 * I)

    def _reset(self):
        if not hasattr(self, "gammas"):
            self.alpha_t_ = self.alpha
            return
        self.expert_alphas_ = np.full(self.gammas.size, self.alpha)
        self.weights_ = np.full(self.gammas.size, 1.0 / self.gammas.size)
        self.alpha_t_ = self.alpha

    def _adapt(self, score) -> bool:
        srt = self._online()
        n = len(srt)
        miss = score > self.threshold()
        r = bisect.bisect_left(srt, score)  # past scores strictly below
        eta, sigma = self._rates
        self.alpha_t_ = _dtaci_step(self.expert_alphas_, self.weights_, self.gammas, self.alpha,
                                    *self._decimal_alpha, eta, sigma, r, n)
        return miss

    def _replay(self, s, use_numba):
        eta, sigma = self._rates
        q, err, alphas, final = dtaci_replay(
            self._scores, s, self.alpha, self.gammas, self.expert_alphas_, self.weights_, eta, sigma,
            self.alpha_t_, self.window or 0, use_numba,
        )
        self.alpha_t_ = final
        return q, err, alphas


class NonexchangeableConformal(AdaptiveConformalInference):
    """Non-exchangeable conformal prediction with recency weights (NexCP;
    Barber, Candes, Ramdas and Tibshirani, 2023).

    Recent scores count more: the score seen ``d`` steps ago has weight
    ``rho ** d``. The threshold is the weighted ``1 - alpha`` quantile of
    the past scores, with the point being predicted counted at +infinity.
    Under drift, old data lose their say gradually, without a hard window.
    The coverage gap is bounded by how much the data distribution moves,
    weighted the same way; with ``rho=1`` this is split conformal
    prediction on all past scores.

    Same calls as :class:`AdaptiveConformalInference` (``calibrate``,
    ``run``, ``threshold``, ``update``, ``predict_interval``).

    Parameters
    ----------
    estimator : object or None
    alpha : float, default=0.1
    rho : float in (0, 1], default=0.99
        Decay per step.

    Reference
    ---------
    Barber, Candes, Ramdas and Tibshirani (2023). Conformal prediction
    beyond exchangeability. Annals of Statistics 51(2), 816-845.
    """

    def __init__(self, estimator=None, alpha: float = 0.1, rho: float = 0.99):
        super().__init__(estimator, alpha, 0.0, None)
        rho = float(rho)
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must be in (0, 1].")
        self.rho = rho

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in ("estimator", "alpha", "rho")}

    def threshold(self) -> float:
        self._online()
        s = np.asarray(self._arrival[self._head :], dtype=float)
        n = s.shape[0]
        if n == 0:
            return math.inf
        w = self.rho ** np.arange(n, 0, -1.0)  # newest score: rho ** 1
        order = np.argsort(s, kind="stable")
        if self.rho == 1.0:  # equal weights: the exact split-conformal rank
            cum = np.arange(1.0, n + 1.0)
            need = float(_rank(n, self.alpha, self.alpha, *self._decimal_alpha))
        else:
            cum = np.cumsum(w[order])
            need = (1.0 - self.alpha) * (cum[-1] + 1.0) * (1.0 + 1e-12)  # rounding may only widen
        pos = int(np.searchsorted(cum, need, side="left"))
        return math.inf if pos >= n else float(s[order][pos])

    def _adapt(self, score) -> bool:
        return score > self.threshold()

    def _replay(self, s, use_numba):
        q, err = nexcp_replay(self._scores, s, self.alpha, self.rho, use_numba)
        return q, err, np.full(s.shape[0], self.alpha)


_AGACI_GAMMAS = (
    0.0, 5e-6, 5e-5,
    1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
    1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
    1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2,
)  # fmt: skip


class AgACI(AdaptiveConformalInference):
    """Aggregated adaptive conformal inference (Zaffran et al., 2022).

    Runs ACI with many step sizes and combines the thresholds they propose,
    with weights learned online by Bernstein online aggregation (BOA) on the
    pinball loss. Where DtACI combines working levels, AgACI combines the
    thresholds themselves, which is steadier when some experts ask for
    infinite intervals. There is no step size to choose.

    Works on scores (for intervals, the absolute error), so one threshold is
    aggregated at pinball level ``1 - alpha``; the paper aggregates the lower
    and upper bounds separately, which is the same thing for symmetric
    intervals. An expert that asks for an infinite threshold is counted at
    the largest score seen so far.

    Same calls as :class:`AdaptiveConformalInference`.

    Parameters
    ----------
    estimator : object or None
    alpha : float, default=0.1
    gammas : sequence of float or None
        Step sizes of the experts. Default: the paper's 30 values from 0 to
        0.09.
    window : int or None

    References
    ----------
    Zaffran, Feron, Goude, Josse and Dieuleveut (2022). Adaptive conformal
    predictions for time series. ICML.

    Wintenberger (2017). Optimal learning with Bernstein online aggregation.
    Machine Learning 106.
    """

    def __init__(self, estimator=None, alpha: float = 0.1, gammas=None, window: int | None = None):
        super().__init__(estimator, alpha, 0.0, window)
        g = np.asarray(_AGACI_GAMMAS if gammas is None else gammas, dtype=float).ravel()
        if g.size == 0 or not np.all(np.isfinite(g)) or np.any(g < 0):
            raise ValueError("gammas must be finite numbers >= 0.")
        self.gammas = g
        self._reset()

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in ("estimator", "alpha", "gammas", "window")}

    def _reset(self):
        self.alpha_t_ = self.alpha
        if not hasattr(self, "gammas"):
            return
        K = self.gammas.size
        self.expert_alphas_ = np.full(K, self.alpha)
        self._eta_inv2 = np.zeros(K)
        self._R = np.zeros(K)
        self._R_reg = np.zeros(K)
        self.weights_ = np.full(K, 1.0 / K)
        self._cap = float(np.max(self._scores)) if self._scores.size else -math.inf

    def _experts(self, srt):
        """Each expert's own threshold on the current calibration set."""
        n = len(srt)
        out = np.empty(self.gammas.size)
        for k, a in enumerate(self.expert_alphas_):
            if a <= 0.0:
                out[k] = math.inf
                continue
            kk = _rank(n, a, self.alpha, *self._decimal_alpha)
            out[k] = math.inf if kk > n else (-math.inf if kk < 1 else srt[kk - 1])
        return out

    def threshold(self) -> float:
        srt = self._online()
        if not srt:
            return math.inf
        return _agaci_combine(self.alpha, self._experts(srt), self._eta_inv2, self._R_reg, self.weights_, self._cap)

    def _adapt(self, score) -> bool:
        srt = self._sorted if self._sorted is not None else self._online()
        raw = self._experts(srt)
        miss_k = score > raw
        if srt:
            pred = _agaci_combine(self.alpha, raw, self._eta_inv2, self._R_reg, self.weights_, self._cap)
            _agaci_learn(self.alpha, self.gammas, self.expert_alphas_, raw, self._eta_inv2, self._R, self._R_reg,
                         score, pred, miss_k)
        else:
            pred = math.inf
            self.expert_alphas_ += self.gammas * (self.alpha - miss_k)
        self._cap = max(self._cap, score)
        return score > pred

    def _replay(self, s, use_numba):
        q, err, self._cap = agaci_replay(
            self._scores, s, self.alpha, self.gammas, self.expert_alphas_, self._eta_inv2, self._R, self._R_reg,
            self._cap, self.window or 0, use_numba,
        )
        return q, err, np.full(s.shape[0], self.alpha)


class ConformalPID(AdaptiveConformalInference):
    """Conformal PID control (Angelopoulos, Candes and Tibshirani, 2023):
    quantile tracking, with an optional integrator.

    The threshold itself is moved, not a working level: up by
    ``lr * (1 - alpha)`` after a miss and down by ``lr * alpha`` after a
    cover (the P term, "quantile tracking"). On its own this never asks for
    an infinite interval and needs no stored scores. With ``KI > 0`` an
    integrator adds a correction that grows with the running coverage error
    (the I term) and can become infinite when coverage has drifted far.
    The D term of the paper, a forecast of the next score, is not included;
    forecast the scores yourself and add the forecast to the threshold.

    Long-run coverage converges to ``1 - alpha`` for any sequence with
    bounded scores.

    Parameters
    ----------
    estimator : object or None
    alpha : float, default=0.1
    lr : float, default=0.1
        Step size. With ``proportional_lr`` it is multiplied by the range of
        the last ``burn_in`` scores, so it adapts to the scale of the data.
        When two or more recent scores are held and all are equal, the
        range is zero and the step would vanish, leaving the threshold stuck
        (for example on a constant series); the largest absolute recent
        score is used instead, and 1 if that is zero too. This departs from
        the authors' code only in that case.
    KI : float, default=0.0
        Integrator gain; 0 turns the integrator off.
    Csat : float, default=1.0
        Integrator saturation constant.
    burn_in : int, default=100
    proportional_lr : bool, default=True

    ``calibrate`` replays the calibration scores through the controller, so
    the threshold starts warmed up.
    """

    def __init__(self, estimator=None, alpha: float = 0.1, lr: float = 0.1, KI: float = 0.0, Csat: float = 1.0,
                 burn_in: int = 100, proportional_lr: bool = True):
        super().__init__(estimator, alpha, 0.0, None)
        if not lr > 0 or burn_in < 1 or not Csat > 0 or KI < 0:
            raise ValueError("lr and Csat must be positive, burn_in at least 1 and KI non-negative.")
        self.lr, self.KI, self.Csat = float(lr), float(KI), float(Csat)
        self.burn_in = int(burn_in)
        self.proportional_lr = bool(proportional_lr)
        self._reset()

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in ("estimator", "alpha", "lr", "KI", "Csat", "burn_in", "proportional_lr")}

    def _reset(self):
        self.alpha_t_ = self.alpha
        if not hasattr(self, "lr"):
            return
        # [threshold part q, integrator, misses so far, steps so far, recent scores held]
        self._state = np.zeros(5)
        self._recent = np.zeros(self.burn_in)
        if getattr(self, "_scores", None) is not None and self._scores.size:
            warm, self._scores = self._scores, np.empty(0)
            self._replay_pid(warm, None)

    def calibrate(self, X=None, y=None, *, y_pred=None, scores=None):
        """Warm the controller up on past scores (optional)."""
        out = super().calibrate(X, y, y_pred=y_pred, scores=scores)
        self._sorted = None
        return out

    def threshold(self) -> float:
        return float(self._state[0] + self._state[1])

    def _replay_pid(self, s, use_numba):
        return pid_replay(
            np.ascontiguousarray(s, dtype=float), self.alpha, self.lr, self.KI, self.Csat, self.burn_in,
            self.proportional_lr, self._state, self._recent, use_numba,
        )

    def update(self, X=None, y=None, *, y_pred=None, score=None) -> bool:
        if score is None:
            score = self._stream(X, y, y_pred, None)[0]
            if score.shape[0] != 1:
                raise ValueError("update() takes one step; use run() for a stream.")
            score = score[0]
        elif X is not None or y is not None or y_pred is not None:
            raise TypeError("Pass score= alone, or a prediction with y.")
        if not math.isfinite(float(score)):
            raise ValueError("score must be finite.")
        return bool(self._replay_pid(np.array([float(score)]), False)[1][0])

    def run(self, X=None, y=None, *, y_pred=None, scores=None, use_numba=None) -> ACIResult:
        """Replay a whole stream; same result as stepping through it."""
        s, p = self._stream(X, y, y_pred, scores)
        q, err = self._replay_pid(s, use_numba)
        res = ACIResult(q, err, np.full(s.shape[0], self.alpha))
        if p is not None:
            res.intervals = _band(p, q)
        return res
