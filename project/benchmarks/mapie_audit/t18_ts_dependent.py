"""Time series on dependent data. EnbPI promises only approximate coverage under conditions, so these are reported, not judged against a bound.
ACI is judged against its bound. Options tested: with and without update, mean and median aggregation, optimize_beta, several levels."""
import numpy as np, warnings
warnings.simplefilter("ignore")
from common import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import PredefinedSplit
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
def series(kind, n, rng):
    e = rng.normal(size=n)
    if kind == "AR(1), phi=0.8, stationary":
        y = np.zeros(n)
        for t in range(1, n): y[t] = 0.8 * y[t - 1] + e[t]
    elif kind == "AR(1) with volatility clusters (GARCH-like)":
        y = np.zeros(n); s = np.ones(n)
        for t in range(1, n): s[t] = np.sqrt(0.1 + 0.85 * s[t - 1] ** 2 * (0.2 + 0.8 * e[t - 1] ** 2) ); y[t] = 0.5 * y[t - 1] + s[t] * e[t]
    elif kind == "trend plus noise that doubles halfway":
        y = 0.01 * np.arange(n) + e * np.where(np.arange(n) > n * 0.6, 2.0, 1.0)
    return y
def lagged(y, p=3):
    X = np.column_stack([y[p - k - 1: len(y) - k - 1] for k in range(p)]); return X, y[p:]
def enbpi(X, y, n, update=True, agg="mean", **pk):
    m = TimeSeriesRegressor(LinearRegression(), method="enbpi", cv=BlockBootstrap(n_resamplings=20, n_blocks=10, overlapping=False, random_state=0), agg_function=agg, n_jobs=1)
    m.fit(X[:n], y[:n]); hit = []; w = []
    for t in range(n, len(y)):
        _, iv = m.predict(X[t:t + 1], confidence_level=0.9, ensemble=True, **pk)
        hit.append(iv[0, 0, 0] <= y[t] <= iv[0, 1, 0]); w.append(iv[0, 1, 0] - iv[0, 0, 0])
        if update: m.update(X[t:t + 1], y[t:t + 1], ensemble=True)
    return np.mean(hit), np.mean(w)
def aci(X, y, n, gamma=0.05):
    m = TimeSeriesRegressor(LinearRegression(), method="aci", cv=BlockBootstrap(n_resamplings=20, n_blocks=10, overlapping=False, random_state=0), agg_function="mean", n_jobs=1)
    m.fit(X[:n], y[:n]); hit = []
    for t in range(n, len(y)):
        _, iv = m.predict(X[t:t + 1], confidence_level=0.9, ensemble=True, allow_infinite_bounds=True)
        hit.append(iv[0, 0, 0] <= y[t] <= iv[0, 1, 0])
        m.adapt_conformal_inference(X[t:t + 1], y[t:t + 1], gamma=gamma, confidence_level=0.9, ensemble=True); m.update(X[t:t + 1], y[t:t + 1], ensemble=True)
    return np.mean(hit)
n, T, S = 400, 600, 6
bound = (0.9 + 0.05) / (0.05 * T)
for kind in ("AR(1), phi=0.8, stationary", "AR(1) with volatility clusters (GARCH-like)", "trend plus noise that doubles halfway"):
    res = {k: [] for k in ("EnbPI with update", "EnbPI without update", "EnbPI median aggregation", "EnbPI optimize_beta", "ACI")}; wid = {}
    for s in range(S):
        rng = np.random.default_rng(100 + s); X, y = lagged(series(kind, n + T + 3, rng))
        for k, f in (("EnbPI with update", lambda: enbpi(X, y, n)), ("EnbPI without update", lambda: enbpi(X, y, n, update=False)), ("EnbPI median aggregation", lambda: enbpi(X, y, n, agg="median")), ("EnbPI optimize_beta", lambda: enbpi(X, y, n, optimize_beta=True))):
            try: c, w = f(); res[k].append(c); wid.setdefault(k, []).append(w)
            except Exception as e: res[k].append(f"{type(e).__name__}: {' '.join(str(e).split())[:90]}")
        try: res["ACI"].append(aci(X, y, n))
        except Exception as e: res["ACI"].append(f"{type(e).__name__}: {' '.join(str(e).split())[:90]}")
    print(f"\n{kind}  (90% target, {S} series of {T} steps)")
    for k, v in res.items():
        nums = [x for x in v if not isinstance(x, str)]; errs = [x for x in v if isinstance(x, str)]
        if errs: print(f"  {k:26s} ERROR {errs[0]}"); continue
        extra = f"  mean width {np.mean(wid[k]):.2f}" if k in wid else f"  bound allows a miss rate up to {0.1 + bound:.3f}: {'VIOLATED in ' + str(sum(1 - x > 0.1 + bound for x in nums)) + ' series' if any(1 - x > 0.1 + bound for x in nums) else 'met in all series'}"
        print(f"  {k:26s} coverage mean {np.mean(nums):.3f}, lowest {min(nums):.3f}, highest {max(nums):.3f}{extra}", flush=True)
