"""Recalibrators compared in compare_*.py. All work per target; M targets and K levels at once.

PIT scale (output: thresholds theta at which the base distribution is read):
  PITTracker (adaptive_cpd.py)   one tracker per level; "sorted" or "independent"
  LazyPIT                        MultiQT's lazy update with isotonic projection (Ding, Gibbs, Tibshirani, arXiv:2512.23671),
                                 run on the PIT scale instead of the outcome scale
  EmpiricalPIT                   online recalibration, G_t = empirical distribution of past PITs. This is Algorithm 1 of
                                 Deshpande, Marx and Kuleshov (arXiv:2302.12196) with the running frequency as the binary
                                 subroutine: a simplification, their theory asks for a calibrated subroutine.
  Jumper                         Vovk's enhancement by betting (arXiv:2105.08669): Simple Jumper, J = 0.01, bets 1 + e (u - 0.5), e in {-1, 0, 1}
Outcome scale:
  MultiQT                        as published: offsets added to the base quantiles, lazy update, isotonic projection,
                                 learning rate max(0.1 * 90% quantile of the last 50 absolute errors over all levels, 0.1)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def pava(X):
    """Isotonic regression (least squares, equal weights) of each row of X."""
    M, K = X.shape; out = np.empty_like(X); val = np.empty(K); wt = np.empty(K)
    for m in range(M):
        n = 0
        for k in range(K):
            val[n] = X[m, k]; wt[n] = 1.0; n += 1
            while n > 1 and val[n - 2] > val[n - 1]:
                w = wt[n - 2] + wt[n - 1]; val[n - 2] = (val[n - 2] * wt[n - 2] + val[n - 1] * wt[n - 1]) / w; wt[n - 2] = w; n -= 1
        k = 0
        for b in range(n):
            for _ in range(int(wt[b])): out[m, k] = val[b]; k += 1
    return out


class LazyPIT:
    def __init__(self, levels, gamma=0.01, n_targets=1):
        self.levels = np.asarray(levels, float); self.gamma = gamma; self.hidden = np.repeat(self.levels[None], n_targets, 0)
    def thresholds(self): return pava(self.hidden)
    def update(self, pit, revealed=True):
        pit = np.atleast_1d(np.asarray(pit, float)); rev = np.broadcast_to(np.atleast_1d(revealed), pit.shape) & ~np.isnan(pit)
        step = self.gamma * (self.levels[None] - (pit[:, None] <= pava(self.hidden)))        # gradient at the played point
        self.hidden = np.where(rev[:, None], self.hidden + step, self.hidden)


class EmpiricalPIT:
    def __init__(self, levels, n_targets=1):
        self.levels = np.asarray(levels, float); self.hist = [[] for _ in range(n_targets)]
    def thresholds(self):
        out = np.repeat(self.levels[None], len(self.hist), 0)
        for m, h in enumerate(self.hist):
            if len(h) >= 10: out[m] = np.quantile(np.asarray(h), self.levels)
        return out
    def update(self, pit, revealed=True):
        pit = np.atleast_1d(np.asarray(pit, float)); rev = np.broadcast_to(np.atleast_1d(revealed), pit.shape)
        for m in range(len(self.hist)):
            if rev[m] and not np.isnan(pit[m]): self.hist[m].append(pit[m])


class Jumper:
    def __init__(self, levels, n_targets=1, J=0.01):
        self.levels = np.asarray(levels, float); self.J = J; self.C = np.full((n_targets, 3), 1 / 3); self.eps = np.array([-1.0, 0.0, 1.0])
    def _mix(self): return (1 - self.J) * self.C + (self.J / 3) * self.C.sum(1, keepdims=True)
    def thresholds(self):
        C = self._mix(); e = (C * self.eps).sum(1) / C.sum(1)                                # mean bet; protected CDF is B(v) = v + e (v^2 - v) / 2
        a = e[:, None] / 2; tau = self.levels[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(np.abs(a) < 1e-12, tau, (-(1 - a) + np.sqrt((1 - a) ** 2 + 4 * a * tau)) / (2 * a))
        return v
    def update(self, pit, revealed=True):
        pit = np.atleast_1d(np.asarray(pit, float)); rev = np.broadcast_to(np.atleast_1d(revealed), pit.shape) & ~np.isnan(pit)
        C = self._mix() * (1 + self.eps[None] * (np.nan_to_num(pit, nan=0.5)[:, None] - 0.5)); C = C / C.sum(1, keepdims=True)
        self.C = np.where(rev[:, None], C, self.C)


class MultiQT:
    def __init__(self, levels, n_targets=1, eps=0.1):
        self.levels = np.asarray(levels, float); self.hidden = np.zeros((n_targets, self.levels.size)); self.eps = eps; self.err = [[] for _ in range(n_targets)]
    def forecast(self, base):
        self._q = pava(base + self.hidden); return self._q
    def update(self, base, y, revealed=True):
        y = np.atleast_1d(np.asarray(y, float)); rev = np.broadcast_to(np.atleast_1d(revealed), y.shape) & ~np.isnan(y)
        for m in range(y.size):
            if not rev[m]: continue
            h = self.err[m]; eta = max(0.1 * np.quantile(np.concatenate(h[-50:]), 0.9), self.eps) if h else self.eps
            self.hidden[m] -= eta * ((y[m] <= self._q[m]) - self.levels); h.append(np.abs(y[m] - base[m]))
