"""PUNCC 0.9.3 regression: coverage on i.i.d. data, exact rank, small calibration sets, weights."""
import sys, math, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from deel.puncc.regression import SplitCP, LocallyAdaptiveCP, CQR, CVPlus, EnbPI, AdaptiveEnbPI, LeverageWeightedCP
from deel.puncc.api.prediction import BasePredictor, DualPredictor, MeanVarPredictor, IdPredictor
R = int(sys.argv[1])
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
def rep(label, f, reps, target=0.9):
    cov = []; inf = 0; err = None
    for r in range(reps):
        try: lo, hi, y = f(r)
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"; break
        lo, hi = np.asarray(lo).ravel(), np.asarray(hi).ravel(); cov.append(((lo <= y) & (y <= hi)).mean()); inf += np.isinf(hi).mean()
    if err: print(f"  {label:62s} ERROR {err}", flush=True); return
    m, se = np.mean(cov), np.std(cov) / np.sqrt(len(cov))
    print(f"  {label:62s} coverage {m:.4f} (se {se:.4f}) {verdict(m, se, target)}; infinite {inf/reps:.2f}", flush=True)
print("(a) fixed predictions (IdPredictor), i.i.d. noise")
for n in (9, 19, 49, 199, 1000):
    for alpha in (0.1, 0.05):
        def f(r, n=n, alpha=alpha):
            g = np.random.default_rng(13 * r + n); y = g.normal(size=n + 500); m = SplitCP(IdPredictor(), train=False); m.fit(X_calib=np.zeros(n), y_calib=y[:n]); _, lo, hi = m.predict(np.zeros(500), alpha=alpha); return lo, hi, y[n:]
        rep(f"SplitCP n={n}, alpha={alpha}", f, R, 1 - alpha)
def f_la(r, n=300):
    g = np.random.default_rng(r); s = np.exp(g.normal(0, 0.5, n + 500)); y = g.normal(size=n + 500) * s; P = np.column_stack([np.zeros(n + 500), s])
    m = LocallyAdaptiveCP(IdPredictor(), train=False); m.fit(X_calib=P[:n], y_calib=y[:n]); _, lo, hi = m.predict(P[n:], alpha=0.1); return lo, hi, y[n:]
rep("LocallyAdaptiveCP, n=300 (predictions: mean and spread)", f_la, R)
def f_cqr(r, n=300, nt=500):
    g = np.random.default_rng(r); s = np.exp(g.normal(0, 0.5, n + nt)); y = g.normal(size=n + nt) * s; P = np.column_stack([-s, s])
    m = CQR(IdPredictor(), train=False); m.fit(X_calib=P[:n], y_calib=y[:n]); _, lo, hi = m.predict(P[n:], alpha=0.1); return lo, hi, y[n:]
rep("CQR, n=300 (predictions: two quantiles)", f_cqr, R)
print("(b) with models, fresh test data, 90%")
def data(g, n):
    X = g.uniform(-2, 2, (n, 4)); y = np.sin(2 * X[:, 0]) + X[:, 1] + g.normal(size=n) * (0.3 + np.abs(X[:, 2])); return X, y
reps = max(20, R // 10)
def w(build):
    def f(r):
        g = np.random.default_rng(500 + r); X, y = data(g, 600); Xt, yt = data(g, 1500); out = build(X, y, Xt, r); return out[1], out[2], yt
    return f
rep("SplitCP + LinearRegression, fit_ratio 0.8", w(lambda X, y, Xt, r: (lambda m: (m.fit(X=X, y=y, fit_ratio=0.8), m.predict(Xt, alpha=0.1))[1])(SplitCP(BasePredictor(LinearRegression()), random_state=r))), reps)
rep("SplitCP + RandomForest, explicit split", w(lambda X, y, Xt, r: (lambda m: (m.fit(X_fit=X[:400], y_fit=y[:400], X_calib=X[400:], y_calib=y[400:]), m.predict(Xt, alpha=0.1))[1])(SplitCP(BasePredictor(RandomForestRegressor(n_estimators=50, random_state=r))))), reps)
rep("LocallyAdaptiveCP + two forests", w(lambda X, y, Xt, r: (lambda m: (m.fit(X=X, y=y, fit_ratio=0.7), m.predict(Xt, alpha=0.1))[1])(LocallyAdaptiveCP(MeanVarPredictor([RandomForestRegressor(n_estimators=50, random_state=r), RandomForestRegressor(n_estimators=50, random_state=r + 1)]), random_state=r))), reps)
rep("CQR + two gradient-boosting quantile models", w(lambda X, y, Xt, r: (lambda m: (m.fit(X=X, y=y, fit_ratio=0.7), m.predict(Xt, alpha=0.1))[1])(CQR(DualPredictor([GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=60, random_state=r), GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=60, random_state=r)]), random_state=r))), reps)
rep("CVPlus, K=5, LinearRegression (guarantee: at least 80%)", w(lambda X, y, Xt, r: (lambda m: (m.fit(X, y), m.predict(Xt, alpha=0.1))[1])(CVPlus(BasePredictor(LinearRegression()), K=5, random_state=r))), reps)
rep("CVPlus, K=5, RandomForest", w(lambda X, y, Xt, r: (lambda m: (m.fit(X, y), m.predict(Xt, alpha=0.1))[1])(CVPlus(BasePredictor(RandomForestRegressor(n_estimators=30, random_state=r)), K=5, random_state=r))), reps)
rep("CVPlus, K=20, 60 training rows, LinearRegression", lambda r: (lambda g: (lambda X, y, Xt, yt: (lambda m: (m.fit(X, y), m.predict(Xt, alpha=0.1))[1][1:] + (yt,))(CVPlus(BasePredictor(LinearRegression()), K=20, random_state=r)))(*data(g, 60), *data(g, 1500)))(np.random.default_rng(700 + r)), reps * 3)
rep("LeverageWeightedCP + LinearRegression", w(lambda X, y, Xt, r: (lambda m: (m.fit(X_fit=X[:400], y_fit=y[:400], X_calib=X[400:], y_calib=y[400:]), m.predict(Xt, alpha=0.1))[1])(LeverageWeightedCP(BasePredictor(LinearRegression())))), reps)
print("(c) exact rank: residuals 1..n, upper end of the interval against ceil((n+1)(1-alpha))")
for alpha in (0.2, 0.1, 0.05, 0.025, 0.01):
    c = [0, 0, 0, 0]; ex = []
    for n in range(1, 2001):
        k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
        if k > n: continue
        try:
            m = SplitCP(IdPredictor(), train=False); m.fit(X_calib=np.zeros(n), y_calib=np.arange(1.0, n + 1)); hi = float(np.asarray(m.predict(np.zeros(1), alpha=alpha)[2]).ravel()[0])
        except Exception as e: c[3] += 1; ex.append((n, type(e).__name__)) if len(ex) < 5 else None; continue
        i = 0 if hi == k else (1 if hi > k else 2); c[i] += 1
        if i and len(ex) < 6: ex.append((n, hi, k))
    print(f"  alpha {alpha}: exact {c[0]}, too high {c[1]}, TOO LOW {c[2]}, refused although a finite rank exists {c[3]}; examples {ex}")
print("(d) small calibration sets, alpha = 0.1")
for n in (1, 2, 5, 8, 9, 10):
    try:
        m = SplitCP(IdPredictor(), train=False); m.fit(X_calib=np.zeros(n), y_calib=np.arange(1.0, n + 1)); out = m.predict(np.zeros(2), alpha=0.1); print(f"  n={n:2d}: interval {np.asarray(out[1]).ravel()[0]}, {np.asarray(out[2]).ravel()[0]}")
    except Exception as e: print(f"  n={n:2d}: {type(e).__name__}: {' '.join(str(e).split())[:150]}")
print("(e) weights (weight_func): exponential-decay weights on i.i.d. data must still cover; all-equal weights must equal no weights")
def f_w(r, wf):
    g = np.random.default_rng(r); n = 300; y = g.normal(size=n + 500); m = SplitCP(IdPredictor(), train=False, weight_func=wf); m.fit(X_calib=np.zeros(n), y_calib=y[:n]); _, lo, hi = m.predict(np.zeros(500), alpha=0.1); return lo, hi, y[n:]
for label, wf in (("equal weights", lambda X: np.ones(len(X))), ("decay 0.99 per step", lambda X: 0.99 ** np.arange(len(X), 0, -1)), ("random positive weights", lambda X: np.random.default_rng(len(X)).random(len(X)) + 0.1)):
    rep(f"SplitCP with weight_func: {label}", lambda r, wf=wf: f_w(r, wf), R)
g = np.random.default_rng(0); y = g.normal(size=300)
a = SplitCP(IdPredictor(), train=False); a.fit(X_calib=np.zeros(300), y_calib=y); b = SplitCP(IdPredictor(), train=False, weight_func=lambda X: np.ones(len(X))); b.fit(X_calib=np.zeros(300), y_calib=y)
try: print("  equal weights give the same interval as no weights:", bool(np.allclose(np.asarray(a.predict(np.zeros(3), alpha=0.1)[2]), np.asarray(b.predict(np.zeros(3), alpha=0.1)[2]))), np.asarray(a.predict(np.zeros(1), alpha=0.1)[2]).ravel(), np.asarray(b.predict(np.zeros(1), alpha=0.1)[2]).ravel())
except Exception as e: print("  ERROR", type(e).__name__, str(e)[:150])
