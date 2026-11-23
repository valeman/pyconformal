"""PUNCC 0.9.3: EnbPI, class-wise LAC, anomaly detection, object detection, metrics, edge cases of APS."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from deel.puncc.regression import EnbPI, AdaptiveEnbPI
from deel.puncc.classification import LAC, ClasswiseLAC, APS, RAPS
from deel.puncc.anomaly_detection import SplitCAD
from deel.puncc.object_detection import SplitBoxWise
from deel.puncc.api.prediction import BasePredictor, IdPredictor, MeanVarPredictor
from deel.puncc import metrics
R = int(sys.argv[1])
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
print("(a) EnbPI on i.i.d. data, linear model, 90%")
def enb(cls, r, s, T=300, n=200, B=20, pred=None):
    g = np.random.default_rng(r); X = g.normal(size=(n + T, 3)); y = X @ np.array([1.0, -1.0, 0.5]) + g.normal(size=n + T)
    m = cls(pred() if pred else BasePredictor(LinearRegression()), B=B, agg_func_loo=np.mean, random_state=r); m.fit(X[:n], y[:n])
    out = m.predict(X[n:], alpha=0.1, y_true=y[n:], s=s) if s is not None else m.predict(X[n:], alpha=0.1)
    lo, hi = np.asarray(out[1]).ravel(), np.asarray(out[2]).ravel(); return ((lo <= y[n:]) & (y[n:] <= hi)).mean()
for cls, name, pred in ((EnbPI, "EnbPI", None), (AdaptiveEnbPI, "AdaptiveEnbPI", lambda: MeanVarPredictor([LinearRegression(), LinearRegression()]))):
    for s in (None, 1, 2, 10, 50):
        cov = []; err = None
        for r in range(max(15, R // 10)):
            try: cov.append(enb(cls, r, s, pred=pred))
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:120]}"; break
        lab = f"{name}, " + ("no feedback" if s is None else f"residuals updated every {s} step(s)")
        if err: print(f"  {lab:52s} ERROR {err}"); continue
        m_, se = np.mean(cov), np.std(cov) / np.sqrt(len(cov)); print(f"  {lab:52s} coverage {m_:.4f} (se {se:.4f}) {verdict(m_, se, 0.9)}", flush=True)
print("(b) class-wise LAC: coverage inside each class, 4 classes of very different difficulty, 90%")
K = 4
def cdraw(g, n):
    y = g.choice(K, n, p=[0.55, 0.3, 0.1, 0.05]); conc = np.array([8.0, 3.0, 1.2, 0.6])[y]; P = np.full((n, K), 1.0)
    P[np.arange(n), y] += conc; P = np.array([g.dirichlet(p) for p in P]); return P, y
for name, make in (("LAC", lambda: LAC(IdPredictor(), train=False)), ("ClasswiseLAC", lambda: ClasswiseLAC(IdPredictor(), train=False))):
    per = [[] for _ in range(K)]; allc = []; err = None
    for r in range(max(30, R // 5)):
        g = np.random.default_rng(r); Pc, yc = cdraw(g, 1500); Pt, yt = cdraw(g, 3000)
        try: m = make(); m.fit(X_calib=Pc, y_calib=yc); sets = m.predict(Pt, alpha=0.1)[-1]
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"; break
        hit = np.array([yt[i] in sets[i] for i in range(len(yt))]); allc.append(hit.mean())
        for c in range(K): per[c].append(hit[yt == c].mean())
    if err: print(f"  {name}: ERROR {err}"); continue
    print(f"  {name:13s}: overall {np.mean(allc):.4f}; per class {[round(float(np.mean(p)), 3) for p in per]}")
print("(c) anomaly detection (SplitCAD): share of NORMAL points flagged must be at most alpha")
for alpha in (0.1, 0.05, 0.01):
    fa = []; det = []; err = None
    for r in range(max(30, R // 5)):
        g = np.random.default_rng(r); z = g.normal(size=(1500, 3))
        try:
            m = SplitCAD(BasePredictor(IsolationForest(n_estimators=50, random_state=r)), train=True, random_state=r); m.fit(z=z[:1000], fit_ratio=0.5)
            fa.append(np.mean(np.asarray(m.predict(z[1000:], alpha=alpha)))); det.append(np.mean(np.asarray(m.predict(g.normal(size=(300, 3)) * 3.5, alpha=alpha))))
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"; break
    if err: print(f"  alpha {alpha}: ERROR {err}"); continue
    m_, se = np.mean(fa), np.std(fa) / np.sqrt(len(fa)); print(f"  alpha {alpha}: normal points flagged {m_:.4f} (se {se:.4f}) {'ABOVE alpha' if m_ > alpha + 3 * se else 'ok'}; true anomalies flagged {np.mean(det):.3f}")
print("(d) object detection (SplitBoxWise): the outer box must contain the true box (all four sides) in at least 90% of cases")
for method in ("additive", "multiplicative"):
    cov = []; err = None
    for r in range(max(30, R // 5)):
        g = np.random.default_rng(r); n = 1300; c = g.uniform(100, 400, (n, 2)); wh = g.uniform(20, 120, (n, 2)); T = np.column_stack([c - wh / 2, c + wh / 2]); Pd = T + g.normal(0, 6, (n, 4)) * g.uniform(0.5, 2, (n, 1))
        try:
            m = SplitBoxWise(IdPredictor(), train=False, method=method); m.fit(X_calib=Pd[:300], y_calib=T[:300]); out = m.predict(Pd[300:], alpha=0.1)
            outer = np.asarray(out[2]); cov.append(np.mean((outer[:, 0] <= T[300:, 0]) & (outer[:, 1] <= T[300:, 1]) & (outer[:, 2] >= T[300:, 2]) & (outer[:, 3] >= T[300:, 3])))
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"; break
    if err: print(f"  {method}: ERROR {err}"); continue
    m_, se = np.mean(cov), np.std(cov) / np.sqrt(len(cov)); print(f"  {method:14s}: {m_:.4f} (se {se:.4f}) {verdict(m_, se, 0.9)}")
print("(e) metrics against hand calculations")
g = np.random.default_rng(1); y = g.normal(size=400); lo = y - g.gamma(2, 0.4, 400); hi = lo + g.gamma(4, 0.5, 400)
ok = lambda a, b: "ok" if np.isclose(a, b) else f"DIFFERS (PUNCC {a:.5f}, by hand {b:.5f})"
cov = ((lo <= y) & (y <= hi)).mean()
print("  regression_mean_coverage", ok(metrics.regression_mean_coverage(y, lo, hi), cov), "| regression_sharpness", ok(metrics.regression_sharpness(lo, hi), (hi - lo).mean()), "| regression_ace", ok(metrics.regression_ace(y, lo, hi, 0.1), cov - 0.9))
yc = g.integers(0, 5, 300); sets = [list(np.flatnonzero(g.random(5) < 0.5)) for _ in range(300)]
print("  classification_mean_coverage", ok(metrics.classification_mean_coverage(yc, sets), np.mean([yc[i] in sets[i] for i in range(300)])), "| classification_mean_size", ok(metrics.classification_mean_size(sets), np.mean([len(s) for s in sets])))
print("(f) APS when the threshold reaches 1 (the model always ranks the true class last in calibration)")
P = np.tile(np.array([[0.5, 0.3, 0.15, 0.05]]), (300, 1)); yb = np.full(300, 3)
for rand in (True, False):
    try:
        m = APS(IdPredictor(), train=False, rand=rand); m.fit(X_calib=P[:200], y_calib=yb[:200]); s = m.predict(P[200:], alpha=0.1)[-1]
        print(f"  rand={rand}: set sizes {sorted(set(len(x) for x in s))}; true class covered in {np.mean([3 in x for x in s]):.3f}")
    except Exception as e: print(f"  rand={rand}: {type(e).__name__}: {' '.join(str(e).split())[:160]}")
