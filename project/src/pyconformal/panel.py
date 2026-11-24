"""Online conformal prediction for panel data: W-TQA and TQA.

A panel is many units (stores, households, stocks, patients) observed over
the same rounds. At each round the outcomes of the calibration units are
known and a prediction set is wanted for each target unit, whose own
outcome arrives late, sometimes, or never.

W-TQA (weighted temporal quantile adjustment) calibrates on the current
round's cross-section and combines two corrections:

* spatial: calibration units are weighted by how close their running
  feature mean is to the target's (Gaussian kernel with bandwidth ``h``);
* temporal: the miscoverage level follows the adaptive conformal update
  ``alpha_t = alpha_{t-1} + gamma (alpha - miss_{t-1})`` on the rounds where
  the target's outcome was revealed. The level is not clipped.

``bandwidth=inf`` gives equal weights, which is TQA (the TQA-E rule of Lin
et al.); ``gamma=0`` keeps the level fixed (the spatial part alone); both
together give split conformal prediction on each round's cross-section.

References
----------
Tu and Giesecke (2026). Online conformal prediction for non-exchangeable
panel data. arXiv:2605.17705 (Algorithm 1).
Lin, Trivedi and Sun (2022). Conformal prediction intervals with temporal
dependence. TMLR.
Barber, Candes, Ramdas and Tibshirani (2023). Conformal prediction beyond
exchangeability. Annals of Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._panel_kernel import HAVE_NUMBA, round_thresholds
from ._quantile import _check_alpha
from .multiqt import MultiQT, QuantileResult

__all__ = ["WeightedTQA", "PanelResult", "PanelPredictiveSystem"]

_TOL = 1e-12


@dataclass
class PanelResult:
    """Outcome of :meth:`WeightedTQA.run`.

    Attributes
    ----------
    thresholds : ndarray (n_rounds, n_targets)
        Score threshold of each prediction set. ``inf`` is the whole line,
        ``-inf`` the empty set.
    covered : ndarray of bool (n_rounds, n_targets)
        Whether the target's outcome fell in its set. Evaluated on every
        round, revealed or not.
    alphas : ndarray (n_rounds, n_targets)
        The adaptive level at each round, before any finitization.
    fallback : ndarray of bool (n_rounds, n_targets)
        Rounds where the level was below the target's own weight, so the
        set is the largest calibration score (``finite=True``) or the whole
        line (``finite=False``).
    intervals : ndarray (n_rounds, n_targets, 2) or None
        ``prediction -/+ threshold`` when predictions were given.
    """

    thresholds: np.ndarray
    covered: np.ndarray
    alphas: np.ndarray
    fallback: np.ndarray
    intervals: np.ndarray | None = None

    @property
    def coverage(self) -> float:
        """Share of all (round, target) pairs covered."""
        return float(np.nanmean(self._cov()))

    @property
    def unit_coverage(self) -> np.ndarray:
        """Coverage of each target over the rounds."""
        return np.nanmean(self._cov(), axis=0)

    def tail_coverage(self, share: float = 0.1) -> float:
        """Mean coverage of the worst-covered ``share`` of targets."""
        u = np.sort(self.unit_coverage)
        return float(u[: max(1, int(np.ceil(share * u.size)))].mean())

    def _cov(self):
        c = self.covered.astype(float)
        c[~self._valid] = np.nan
        return c

    _valid: np.ndarray = None  # rounds where the target had a score


class WeightedTQA:
    """W-TQA for panel data (Tu and Giesecke, 2026); TQA as a special case.

    Parameters
    ----------
    alpha : float
        Target miscoverage, 0.1 for 90% sets.
    bandwidth : float, default=0.6
        Kernel bandwidth ``h`` on the standardized distance between running
        feature means. Smaller means only very similar units count.
        ``np.inf`` gives equal weights (TQA).
    gamma : float, default=0.01
        Step size of the level update. ``0`` keeps the level at ``alpha``.
    finite : bool, default=False
        When the level falls below the target's own weight the exact
        threshold is infinite. ``False`` returns that infinite set, which
        keeps the bound below for any data. ``True`` replaces it by the
        largest calibration score of the round. That is the paper's
        Algorithm 1 ("finitization"); every set is then bounded, but a
        target that is harder than all of its peers can be missed far more
        often than ``alpha``, and the paper needs an extra assumption
        (its Assumption 5.6) to bound the miss rate.

    Notes
    -----
    Each target is handled on its own: its own weights and its own level.
    Past data enter only through the running feature means and the level,
    so memory does not grow with the number of rounds.

    The cross-sectional standard deviation that scales each feature is taken
    over the calibration units and the target, with divisor ``n`` (the
    paper does not say which divisor it uses).

    Guarantee: on the rounds where the outcome is revealed, the average
    miss rate equals ``alpha + (alpha - alpha_final) / (gamma * n_revealed)``
    exactly (Proposition 5.5). With ``finite=False`` the level cannot fall
    below ``-gamma (1 - alpha)``, so that miss rate is at most
    ``alpha + (alpha + gamma (1 - alpha)) / (gamma * n_revealed)`` for any
    data. Without assumptions linking the target to its peers, no method
    can guarantee coverage at every round.

    Examples
    --------
    >>> tqa = WeightedTQA(alpha=0.1)
    >>> res = tqa.run(y_pred_cal=P_cal, y_cal=Y_cal, X_cal=X_cal,
    ...               y_pred_target=P_tgt, y_target=Y_tgt, X_target=X_tgt)
    >>> res.intervals.shape          # (n_rounds, n_targets, 2)
    """

    def __init__(self, alpha: float = 0.1, bandwidth: float = 0.6, gamma: float = 0.01, finite: bool = False):
        self.alpha = _check_alpha(alpha)
        if not bandwidth > 0:
            raise ValueError("bandwidth must be positive (np.inf for equal weights).")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.bandwidth = float(bandwidth)
        self.gamma = float(gamma)
        self.finite = bool(finite)
        self._started = False

    # ------------------------------------------------------------------ state
    def start(self, n_cal: int, n_targets: int = 1, n_features: int = 0):
        """Reset the state for a panel of ``n_cal`` calibration units."""
        self.n_cal_, self.n_targets_, self.n_features_ = int(n_cal), int(n_targets), int(n_features)
        self.alpha_t_ = np.full(self.n_targets_, self.alpha)
        self._mu_cal = np.zeros((self.n_cal_, self.n_features_))
        self._mu_tgt = np.zeros((self.n_targets_, self.n_features_))
        self._cnt_cal = np.zeros(self.n_cal_)
        self._cnt_tgt = np.zeros(self.n_targets_)
        self._kappa = np.ones((self.n_targets_, self.n_cal_))
        self._geom = None            # centred means and scales, for the compiled kernel
        self.n_revealed_ = np.zeros(self.n_targets_, dtype=int)
        self._started = True
        return self

    def weights(self):
        """Current normalized weights, shape (n_targets, n_cal + 1); the
        last column is the target's own weight."""
        self._need()
        if self._kappa is None:
            self._kappa = self._kappa_numpy()
        k = np.concatenate([self._kappa, np.ones((self.n_targets_, 1))], axis=1)
        return k / k.sum(axis=1, keepdims=True)

    def _need(self):
        if not self._started:
            raise RuntimeError("Call start() or run() first.")

    # ------------------------------------------------------------- one round
    def threshold(self, scores_cal):
        """Thresholds for the current round from this round's calibration
        scores (length ``n_cal``; ``nan`` marks a unit with no outcome this
        round). Returns ``(thresholds, fallback)``, each of length
        ``n_targets``."""
        self._need()
        s = np.asarray(scores_cal, dtype=float).ravel()
        if s.size != self.n_cal_:
            raise ValueError("scores_cal has the wrong length.")
        return self._threshold(s)

    def _threshold(self, s):
        M = self.n_targets_
        ok = ~np.isnan(s)
        q = np.full(M, np.inf)
        if not ok.any():
            return q, np.zeros(M, dtype=bool)
        order = np.argsort(s[ok], kind="stable")
        ss = s[ok][order]
        if self._geom is not None and HAVE_NUMBA:
            A, B, inv, c = self._geom
            fallback = np.zeros(M, dtype=np.bool_)
            round_thresholds(A, B, inv, np.flatnonzero(ok)[order], ss, self.alpha_t_, self.finite, c, _TOL, q, fallback)
            return q, fallback
        if self._kappa is None:
            self._kappa = self._kappa_numpy()
        K = self._kappa[:, ok][:, order]
        cum = np.cumsum(K, axis=1)
        total = cum[:, -1] + 1.0
        w_tgt = 1.0 / total
        a = self.alpha_t_
        fallback = a < w_tgt
        level = np.maximum(a, w_tgt) if self.finite else a
        need = (1.0 - level) * total
        idx = (cum >= (need - _TOL * total)[:, None]).argmax(axis=1)
        found = cum[:, -1] >= need - _TOL * total
        q = np.where(found, ss[idx], np.inf)
        if self.finite:
            q = np.where(fallback, ss[-1], q)
        q = np.where(level >= 1.0, -np.inf, q)
        return q, fallback

    def update(self, X_cal=None, X_target=None, missed=None):
        """Close the round: update the running feature means and, for the
        targets whose outcome was revealed, the level.

        ``missed``: per target, ``True``/``False`` if the outcome was
        revealed and missed/covered, ``None`` or ``nan`` if not revealed.
        """
        self._need()
        if missed is not None:
            m = np.asarray(missed, dtype=float).ravel()
            if m.size == 1 and self.n_targets_ > 1:
                m = np.full(self.n_targets_, m[0])
            rev = ~np.isnan(m)
            self.alpha_t_ = np.where(rev, self.alpha_t_ + self.gamma * (self.alpha - np.nan_to_num(m)), self.alpha_t_)
            self.n_revealed_ += rev
        if self.n_features_ and X_cal is not None:
            self._mean_update(self._mu_cal, self._cnt_cal, np.asarray(X_cal, dtype=float).reshape(self.n_cal_, -1))
            self._mean_update(self._mu_tgt, self._cnt_tgt, np.asarray(X_target, dtype=float).reshape(self.n_targets_, -1))
            self._refresh_kappa()
        return self

    @staticmethod
    def _mean_update(mu, cnt, X):
        ok = ~np.isnan(X).any(axis=1)
        cnt[ok] += 1.0
        mu[ok] += (X[ok] - mu[ok]) / cnt[ok][:, None]

    def _refresh_kappa(self):
        if not np.isfinite(self.bandwidth):
            return
        n, d = self._mu_cal.shape
        c = self._mu_cal.mean(axis=0)
        A = self._mu_cal - c                      # (n, d), columns sum to 0
        B = self._mu_tgt - c                      # (M, d)
        var = (np.sum(A * A, axis=0)[None, :] + B * B) / (n + 1) - (B / (n + 1)) ** 2
        var = np.where(var > 1e-300, var, 1.0)    # zero spread: scale 1
        self._geom = (A, B, 1.0 / var, 1.0 / (2.0 * d * self.bandwidth**2))
        self._kappa = None                        # built only when asked for

    def _kappa_numpy(self):
        A, B, inv, c = self._geom
        D2 = inv @ (A * A).T - 2.0 * (B * inv) @ A.T + np.sum(B * B * inv, axis=1)[:, None]
        np.maximum(D2, 0.0, out=D2)
        return np.exp(-D2 * c)

    # ------------------------------------------------------------------- run
    def run(self, *, scores_cal=None, scores_target=None, X_cal=None, X_target=None, revealed=None,
            y_pred_cal=None, y_cal=None, y_pred_target=None, y_target=None):
        """Replay a panel.

        Give either scores (``scores_cal`` of shape (n_rounds, n_cal) and
        ``scores_target`` of shape (n_rounds, n_targets)), or predictions
        and outcomes, from which absolute-residual scores are formed.

        ``X_cal`` (n_rounds, n_cal, n_features) and ``X_target`` (n_rounds,
        n_targets, n_features) feed the similarity weights; leave them out
        for equal weights. ``revealed``: bool, (n_rounds,) or (n_rounds,
        n_targets); ``True`` where the target's outcome becomes known
        before the next round. Default: always.
        """
        P_t = None
        if scores_cal is None:
            scores_cal = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float))
            P_t = np.asarray(y_pred_target, dtype=float)
            P_t = P_t[:, None] if P_t.ndim == 1 else P_t
            Yt = np.asarray(y_target, dtype=float)
            scores_target = np.abs((Yt[:, None] if Yt.ndim == 1 else Yt) - P_t)
        S = np.asarray(scores_cal, dtype=float)
        St = np.asarray(scores_target, dtype=float)
        St = St[:, None] if St.ndim == 1 else St
        T, N = S.shape
        M = St.shape[1]
        if St.shape[0] != T:
            raise ValueError("scores_cal and scores_target need the same number of rounds.")
        use_x = X_cal is not None and np.isfinite(self.bandwidth)
        if use_x:
            Xc = np.asarray(X_cal, dtype=float).reshape(T, N, -1)
            Xt = np.asarray(X_target, dtype=float).reshape(T, M, -1)
            d = Xc.shape[2]
        else:
            d = 0
        if revealed is None:
            R = np.ones((T, M), dtype=bool)
        else:
            R = np.asarray(revealed, dtype=bool)
            R = np.broadcast_to(R[:, None] if R.ndim == 1 else R, (T, M))
        self.start(N, M, d)
        q = np.empty((T, M)); al = np.empty((T, M)); fb = np.zeros((T, M), dtype=bool)
        for t in range(T):
            al[t] = self.alpha_t_
            q[t], fb[t] = self._threshold(S[t])
            miss = (St[t] > q[t]).astype(float)
            miss[~R[t] | np.isnan(St[t])] = np.nan
            self.update(Xc[t] if use_x else None, Xt[t] if use_x else None, miss)
        valid = ~np.isnan(St)
        res = PanelResult(q, (St <= q) & valid, al, fb,
                          None if P_t is None else np.stack([P_t - q, P_t + q], axis=-1))
        res._valid = valid
        return res


class PanelPredictiveSystem:
    """A predictive distribution for each target unit of a panel.

    Each round, the base distribution of a target is the weighted
    distribution of its peers' signed forecast errors in that round, with
    W-TQA's similarity weights (Tu and Giesecke, 2026). The quantiles of
    that base at ``levels`` are then adjusted online, per target, by MultiQT
    (Ding, Gibbs and Tibshirani, 2025), using the target's own outcome on
    the rounds where it is revealed. The quantiles never cross.

    Parameters
    ----------
    levels : sequence of float, optional
        Probability levels; default 0.05, 0.10, ..., 0.95.
    bandwidth : float, default=0.6
        Kernel bandwidth of the similarity weights; ``np.inf`` for equal weights.
    lr, lr_window, lr_floor
        Step-size settings of :class:`MultiQT`. ``lr_floor`` is in outcome
        units: lower it when forecast errors are much smaller than 1.

    Notes
    -----
    The base quantiles are confined to the range of the peers' errors in the
    round; the adjustment can move outside that range.

    Guarantee: MultiQT's offsets move only on revealed rounds, so its
    calibration result applies to the sequence of revealed rounds of each
    target: with bounded errors, the share of revealed rounds with the
    outcome at or below the level-q quantile tends to q, for every level and
    any data. If which rounds are revealed does not depend on the outcomes,
    the same holds on average over all rounds. Nothing is promised for a
    target that is never revealed, beyond what its peers provide.

    Examples
    --------
    >>> pps = PanelPredictiveSystem()
    >>> res = pps.run(y_pred_cal=P_cal, y_cal=Y_cal, X_cal=X_cal,
    ...               y_pred_target=P_tgt, y_target=Y_tgt, X_target=X_tgt, revealed=seen)
    >>> res.quantiles.shape          # (n_rounds, n_targets, n_levels)
    >>> res.interval(0.1)            # central 90% intervals
    """

    def __init__(self, levels=None, bandwidth: float = 0.6, lr="adaptive", lr_window: int = 50, lr_floor: float = 0.1):
        self.levels = np.round(np.arange(0.05, 0.96, 0.05), 2) if levels is None else np.asarray(levels, dtype=float).ravel()
        self._weights = WeightedTQA(bandwidth=bandwidth)
        self._tracker = MultiQT(self.levels, lr=lr, lr_window=lr_window, lr_floor=lr_floor)
        self.bandwidth = self._weights.bandwidth

    def start(self, n_cal: int, n_targets: int = 1, n_features: int = 0):
        """Reset the state."""
        self._weights.start(n_cal, n_targets, n_features)
        self._tracker.start(n_targets)
        return self

    def base_quantiles(self, resid_cal):
        """Quantiles, at ``levels``, of the weighted distribution of this round's peer errors: (n_targets, n_levels)."""
        r = np.asarray(resid_cal, dtype=float).ravel()
        M, K = self._weights.n_targets_, self.levels.size
        ok = ~np.isnan(r)
        if not ok.any():
            return np.zeros((M, K))
        order = np.argsort(r[ok], kind="stable")
        rs = r[ok][order]
        W = self._weights.weights()
        kap = W[:, :-1][:, ok][:, order]
        tot = kap.sum(axis=1) + W[:, -1]
        cum = np.cumsum(kap, axis=1) / tot[:, None]
        w_tgt = W[:, -1] / tot
        need = self.levels[None, :] - 0.5 * w_tgt[:, None] - _TOL        # half of the target's own mass at each end
        n = rs.size
        cum[:, -1] = np.maximum(cum[:, -1], 1.0 - w_tgt)                 # guard the last cumulative value against rounding
        flat = (cum + 2.0 * np.arange(M)[:, None]).ravel()
        idx = np.searchsorted(flat, (need + 2.0 * np.arange(M)[:, None]).ravel(), side="left").reshape(M, K) - n * np.arange(M)[:, None]
        return rs[np.clip(idx, 0, n - 1)]

    def predict(self, resid_cal):
        """Ordered quantiles of each target's error for this round, and the base they were built from."""
        base = self.base_quantiles(resid_cal)
        return self._tracker.predict(base), base

    def update(self, played, base, resid_target=None, observed=None, X_cal=None, X_target=None):
        """Close the round: feed the targets' own errors where revealed, and the features for the weights."""
        self._tracker.begin_round()
        if resid_target is not None:
            self._tracker.update(resid_target, played, base, observed)
        self._tracker.end_round()
        self._weights.update(X_cal, X_target)
        return self

    def run(self, *, y_pred_cal, y_cal, y_pred_target, y_target, X_cal=None, X_target=None, revealed=None):
        """Replay a panel. Shapes as in :meth:`WeightedTQA.run`. Returns a :class:`QuantileResult` whose
        quantiles are on the outcome scale (target forecast plus error quantile)."""
        Rc = np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float)
        Pt = np.asarray(y_pred_target, dtype=float)
        Pt = Pt[:, None] if Pt.ndim == 1 else Pt
        Yt = np.asarray(y_target, dtype=float)
        Yt = Yt[:, None] if Yt.ndim == 1 else Yt
        Rt = Yt - Pt
        T, N = Rc.shape
        M = Rt.shape[1]
        use_x = X_cal is not None and np.isfinite(self.bandwidth)
        if use_x:
            Xc = np.asarray(X_cal, dtype=float).reshape(T, N, -1)
            Xt = np.asarray(X_target, dtype=float).reshape(T, M, -1)
        R = np.ones((T, M), dtype=bool) if revealed is None else np.asarray(revealed, dtype=bool)
        R = np.broadcast_to(R[:, None] if R.ndim == 1 else R, (T, M))
        self.start(N, M, Xc.shape[2] if use_x else 0)
        Q = np.empty((T, M, self.levels.size))
        B = np.empty_like(Q)
        for t in range(T):
            Q[t], B[t] = self.predict(Rc[t])
            self.update(Q[t], B[t], Rt[t], R[t], Xc[t] if use_x else None, Xt[t] if use_x else None)
        res = QuantileResult(Q + Pt[:, :, None], self.levels, Yt)
        res.base_quantiles = B + Pt[:, :, None]
        return res
