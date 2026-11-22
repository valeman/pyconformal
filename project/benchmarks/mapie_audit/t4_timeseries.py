"""TimeSeriesRegressor: ACI and EnbPI, following MAPIE's documented loops.
ACI's theorem (Gibbs & Candes 2021): on ANY sequence, |miss rate - alpha| <= (max(a1, 1 - a1) + gamma) / (gamma * T)."""
import sys, numpy as np
from common import *
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap

def aci(y, n0, gamma, level=0.9):
    X = np.zeros((len(y), 1))
    m = TimeSeriesRegressor(Const(0.0), method="aci", cv=PredefinedSplit([-1] * 1 + [0] * (n0 - 1)), random_state=0)
    m.fit(X[:n0], y[:n0])
    lo, hi = [], []
    for t in range(n0, len(y)):
        _, iv = m.predict(X[t:t + 1], confidence_level=level, ensemble=False, allow_infinite_bounds=True)
        lo.append(iv[0, 0, 0]); hi.append(iv[0, 1, 0])
        m.adapt_conformal_inference(X[t:t + 1], y[t:t + 1], gamma=gamma, confidence_level=level, ensemble=False)
        m.update(X[t:t + 1], y[t:t + 1], ensemble=False)
    lo, hi = np.array(lo), np.array(hi); yt = y[n0:]
    return ((lo <= yt) & (yt <= hi)), np.isinf(hi).mean(), m

T, n0, gamma, alpha = 3000, 200, 0.05, 0.1
bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
print(f"ACI, gamma={gamma}, T={T}: theorem allows |miss rate - {alpha}| <= {bound:.4f}")
rng = np.random.default_rng(0)
cases = {
    "i.i.d. noise": rng.normal(size=n0 + T),
    "noise scale x10 halfway": np.r_[rng.normal(size=n0 + T // 2), 10 * rng.normal(size=T - T // 2)],
    "errors growing 1% per step": rng.normal(size=n0 + T) * 1.01 ** np.arange(n0 + T),
    "errors growing 5% per step": rng.normal(size=n0 + T) * 1.05 ** np.arange(n0 + T),
    "every outcome larger than all before": np.arange(1.0, n0 + T + 1) ** 2,
}
for name, y in cases.items():
    try:
        cov, inf, m = aci(y, n0, gamma)
        miss = 1 - cov.mean()
        print(f"  {name:40s} miss rate {miss:.4f}  deviation {abs(miss - alpha):.4f}  {'VIOLATES THE BOUND' if abs(miss - alpha) > bound + 1e-12 else 'within bound'}  infinite intervals {inf:.2%}  final alpha_t {getattr(m, 'current_alpha', {})}")
    except Exception as e:
        print(f"  {name:40s} ERROR {type(e).__name__}: {str(e)[:120]}")

# EnbPI on i.i.d. data (the easy case): fit, then predict/update one step at a time
print("EnbPI, i.i.d. linear data, 90% target, documented predict/update loop")
for seed in range(5):
    rng = np.random.default_rng(seed); n, Tt = 300, 400
    X = rng.normal(size=(n + Tt, 3)); y = X @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=n + Tt)
    m = TimeSeriesRegressor(LinearRegression(), method="enbpi", cv=BlockBootstrap(n_resamplings=20, n_blocks=10, overlapping=False, random_state=seed), agg_function="mean", n_jobs=1)
    m.fit(X[:n], y[:n]); hit = []
    for t in range(n, n + Tt):
        _, iv = m.predict(X[t:t + 1], confidence_level=0.9, ensemble=True)
        hit.append(iv[0, 0, 0] <= y[t] <= iv[0, 1, 0]); m.update(X[t:t + 1], y[t:t + 1], ensemble=True)
    print(f"  seed {seed}: coverage {np.mean(hit):.3f}")
