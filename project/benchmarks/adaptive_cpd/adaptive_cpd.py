"""Research prototype (not part of the package API): predictive distributions that stay calibrated under drift.

PITTracker: online recalibration of any predictive distribution through its probability integral transform (PIT).
For each probability level tau_k it keeps a threshold theta_k on the PIT scale, updated on each revealed outcome by

    theta_k <- theta_k + gamma * (tau_k - 1{PIT <= theta_k})                      (no clipping)

The recalibrated tau_k-quantile is the base distribution's theta_k-quantile (minus infinity for theta_k <= 0, plus
infinity for theta_k >= 1). This is the adaptive conformal update of Gibbs and Candes applied at every level at once.

Facts used below (proved in README.md):
  1. For any sequence, |freq{y_t <= q_k,t} - tau_k| <= (1 + gamma) / (gamma * n) at every level k.
  2. Thresholds of two levels can cross, but never by gamma or more, and a crossing is undone at the next update
     that separates them. mode="sorted" returns the thresholds sorted, so the output is always a valid distribution
     and each sorted threshold is within gamma of one with the exact guarantee.
"""
import numpy as np


class PITTracker:
    def __init__(self, levels, gamma=0.01, n_targets=1, mode="sorted"):
        self.levels = np.asarray(levels, dtype=float)
        self.gamma, self.mode = float(gamma), mode
        self.theta = np.repeat(self.levels[None, :], n_targets, axis=0)

    def thresholds(self):
        return np.sort(self.theta, axis=1) if self.mode == "sorted" else self.theta

    def update(self, pit, revealed=True):
        pit = np.atleast_1d(np.asarray(pit, dtype=float))
        rev = np.broadcast_to(np.atleast_1d(revealed), pit.shape) & ~np.isnan(pit)
        step = self.gamma * (self.levels[None, :] - (pit[:, None] <= self.theta))
        self.theta = np.where(rev[:, None], self.theta + step, self.theta)


def rolling_cps(resid_window, theta):
    """Quantiles of a conformal predictive system built on a window of signed residuals, at PIT levels ``theta``.
    Each residual carries mass 1/(W+1); the remaining 1/(W+1) is split between the two ends."""
    r = np.sort(resid_window); W = r.size
    k = np.ceil(theta * (W + 1) - 0.5 - 1e-12).astype(int)
    q = r[np.clip(k, 1, W) - 1].astype(float)
    q[k < 1] = -np.inf; q[k > W] = np.inf
    return q


def rolling_pit(resid_window, r_new):
    return (np.sum(resid_window <= r_new) + 0.5) / (resid_window.size + 1)


def weighted_quantiles(r_sorted, cum, w_tgt, theta):
    """Weighted cross-sectional distribution: ``cum`` (M, N) cumulative normalized weights of the peers in the order
    of ``r_sorted``; the target's own weight ``w_tgt`` (M,) is split between the two ends. ``theta`` (M, K)."""
    need = theta - 0.5 * w_tgt[:, None]
    idx = (cum[:, :, None] >= need[:, None, :] - 1e-12).argmax(axis=1)
    q = r_sorted[idx]
    q = np.where(need > cum[:, -1][:, None] + 1e-12, np.inf, q)
    return np.where(theta <= 0.5 * w_tgt[:, None], -np.inf, q)


def pinball(q, y, levels):
    """Mean pinball loss over levels; q (..., K), y (...)."""
    d = y[..., None] - q
    return np.mean(np.maximum(levels * d, (levels - 1) * d), axis=-1)
