"""Additional experiments for the technical paper (sections: live, va, forest, classes, cir).

Usage: python review_experiments.py live va forest classes cir
"live" needs MAPIE; "va" needs venn-abers; the rest need only pyconformal with numba.
"""
import sys, time, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
import pyconformal as pc
which = set(sys.argv[1:]) or {"live", "va", "forest", "classes", "cir"}
pct = lambda v: "median %.1f, 95th %.1f, 99th %.1f, max %.0f" % (np.median(v), np.percentile(v, 95), np.percentile(v, 99), np.max(v))

if "live" in which:
    # One row at a time, with a stub model of controlled cost, so the time measured is the conformal step.
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.linear_model import LinearRegression
    from mapie.regression import TimeSeriesRegressor
    from mapie.subsample import BlockBootstrap
    class Stub(BaseEstimator, RegressorMixin):
        """Returns the first feature; optionally burns a fixed time per call."""
        def __init__(self, cost=0.0): self.cost = cost
        def fit(self, X, y): self.fitted_ = True; return self
        def predict(self, X):
            if self.cost:
                end = time.perf_counter() + self.cost
                while time.perf_counter() < end: pass
            return np.asarray(X)[:, 0]
    rng = np.random.default_rng(0); T0, T = 500, 3000
    f = np.cumsum(rng.normal(size=T0 + T)) * 0.1; y = f + rng.standard_t(3, T0 + T); X = f[:, None]
    print("LIVE: ACI one row at a time, microseconds per step (predict, then update)")
    for cost in (0.0, 100e-6):
        m = Stub(cost).fit(X[:10], y[:10])
        mp = TimeSeriesRegressor(m, method="aci", cv="prefit"); mp.fit(X[:T0], y[:T0]); lat = np.empty(T)
        for t in range(T):
            a = time.perf_counter(); mp.predict(X[T0 + t:T0 + t + 1], confidence_level=0.9, allow_infinite_bounds=True)
            mp.adapt_conformal_inference(X[T0 + t:T0 + t + 1], y[T0 + t:T0 + t + 1], gamma=0.01, confidence_level=0.9); lat[t] = (time.perf_counter() - a) * 1e6
        aci = pc.AdaptiveConformalInference(m, alpha=0.1, gamma=0.01).calibrate(X[:T0], y[:T0]); lo = np.empty(T)
        for t in range(T):
            a = time.perf_counter(); p = m.predict(X[T0 + t:T0 + t + 1]); aci.predict_interval(y_pred=p); aci.update(y_pred=p, y=y[T0 + t:T0 + t + 1]); lo[t] = (time.perf_counter() - a) * 1e6
        print(f"  model cost {cost * 1e6:.0f} us per call: MAPIE {pct(lat)} | pyconformal {pct(lo)}")
    print("LIVE: EnbPI one row at a time, 30 linear models, microseconds per step")
    Xe = rng.normal(size=(1000 + 500, 5)); ye = Xe @ np.arange(1.0, 6.0) + rng.normal(size=1500)
    mp = TimeSeriesRegressor(LinearRegression(), method="enbpi", agg_function="mean", cv=BlockBootstrap(n_resamplings=30, n_blocks=10, overlapping=False, random_state=0)); mp.fit(Xe[:1000], ye[:1000]); lat = np.empty(500)
    for t in range(500):
        a = time.perf_counter(); mp.predict(Xe[1000 + t:1001 + t], confidence_level=0.9, ensemble=True); mp.update(Xe[1000 + t:1001 + t], ye[1000 + t:1001 + t], ensemble=True); lat[t] = (time.perf_counter() - a) * 1e6
    en = pc.EnbPI(LinearRegression(), alpha=0.1, n_bootstraps=30, random_state=0).fit(Xe[:1000], ye[:1000]); lo = np.empty(500)
    for t in range(500):
        a = time.perf_counter(); en.predict_interval(Xe[1000 + t:1001 + t]); en.update(Xe[1000 + t:1001 + t], ye[1000 + t:1001 + t]); lo[t] = (time.perf_counter() - a) * 1e6
    print(f"  MAPIE {pct(lat)} | pyconformal {pct(lo)}")

if "va" in which:
    from venn_abers import VennAbers as Ref
    def cases(rng, n):
        s = rng.uniform(size=n)
        yield "continuous, informative", s, (rng.uniform(size=n) < s).astype(int)
        yield "pure noise", s, rng.integers(0, 2, n)
        yield "anti-correlated", s, (rng.uniform(size=n) < 1 - s).astype(int)
        yield "perfectly separated", s, (s > 0.5).astype(int)
        o = np.argsort(s); alt = np.empty(n, int); alt[o] = np.arange(n) % 2
        yield "labels alternate in score order", s, alt
        b = max(2, int(np.sqrt(n))); blk = np.empty(n, int); blk[o] = (np.arange(n) // b) % 2
        yield "blocks of sqrt(n) equal labels", s, blk
        yield "ten distinct scores", np.round(s * 9) / 9, (rng.uniform(size=n) < s).astype(int)
        yield "all scores equal", np.full(n, 0.5), rng.integers(0, 2, n)
        g = 1 - 2.0 ** -rng.integers(1, 40, n)
        yield "geometrically spaced scores", g, (rng.uniform(size=n) < g).astype(int)
    def run(s, y, st):
        t = time.perf_counter(); r = pc.VennAbers().calibrate(scores=s, y=y).predict_p0_p1(scores=st); return time.perf_counter() - t, r
    rng = np.random.default_rng(0); run(*list(cases(rng, 1000))[0][1:], rng.uniform(size=100))
    print("VENN-ABERS: stress configurations (seconds for calibration plus n test points; ratio 10 means linear)")
    for (name, s5, y5), (_, s6, y6) in zip(cases(np.random.default_rng(1), 100_000), cases(np.random.default_rng(2), 1_000_000)):
        t5 = min(run(s5, y5, s5)[0] for _ in range(3)); t6 = min(run(s6, y6, s6)[0] for _ in range(2))
        sm, ym = s5[:3000], y5[:3000]; st = np.r_[sm[:500], rng.uniform(size=500)]
        ours = run(sm, ym, st)[1]; ref = Ref(); ref.fit(np.c_[1 - sm, sm], ym); p0p1 = ref.predict_proba(np.c_[1 - st, st])[1]
        print(f"  {name:34} 1e5: {t5:6.3f}  1e6: {t6:6.3f}  ratio {t6 / t5:5.1f}  largest difference from venn-abers {np.abs(np.asarray(ours) - p0p1).max():.1e}")

if "forest" in which:
    print("FOREST: CQR on the forest's 5% and 95% quantiles, exact leaves against one sample per leaf (10 repetitions)")
    res = {"exact": [], "one sample per leaf": []}
    for rep in range(10):
        rng = np.random.default_rng(rep); n = 30_000
        X = rng.uniform(-2, 2, size=(n, 5)); y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + (0.3 + 0.5 * np.abs(X[:, 2])) * rng.standard_t(4, n)
        tr, ca, te = slice(0, 15_000), slice(15_000, 20_000), slice(20_000, n)
        for name, kw in (("exact", {}), ("one sample per leaf", {"max_samples_leaf": 1})):
            fo = pc.QuantileForest(n_estimators=100, random_state=rep, **kw).fit(X[tr], y[tr])
            t = time.perf_counter(); q = fo.predict(X[te], quantiles=[0.05, 0.95]); tp = time.perf_counter() - t
            pin = np.mean([np.mean(np.maximum(a * (y[te] - q[:, j]), (a - 1) * (y[te] - q[:, j]))) for j, a in enumerate((0.05, 0.95))])
            iv = pc.ConformalizedQuantileRegressor(alpha=0.1).calibrate(quantiles=fo.predict(X[ca], quantiles=[0.05, 0.95]), y=y[ca]).predict_interval(quantiles=q)
            res[name].append((np.mean((iv[:, 0] <= y[te]) & (y[te] <= iv[:, 1])), np.mean(iv[:, 1] - iv[:, 0]), pin, np.mean((q[:, 0] <= y[te]) & (y[te] <= q[:, 1])), tp))
    for name, v in res.items():
        v = np.array(v); m, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
        print(f"  {name:20} raw coverage {m[3]:.4f} | after CQR: coverage {m[0]:.4f} ({se[0]:.4f}), length {m[1]:.3f} ({se[1]:.3f}) | pinball {m[2]:.4f} ({se[2]:.4f}) | predict {m[4]:.3f} s")

if "classes" in which:
    import resource
    print("CLASSES: randomised APS, 2,000 calibration and 2,000 test rows (calibrate + predict; best of 3)")
    for K in (100, 1000, 10_000, 30_000):
        rng = np.random.default_rng(K); n = 4000
        yy = rng.integers(0, K, n); lg = (rng.normal(size=(n, K)) * 1.5).astype(np.float64); lg[np.arange(n), yy] += 4.0
        P = np.exp(lg - lg.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); del lg
        b = 1e9
        for _ in range(3):
            t = time.perf_counter(); S = pc.SplitConformalClassifier(alpha=0.1, score="aps", random_state=0, classes=np.arange(K)).calibrate(proba=P[:2000], y=yy[:2000]).predict_set(proba=P[2000:]); b = min(b, time.perf_counter() - t)
        print(f"  {K:6} classes: {b:7.3f} s  ({b / (2000 * K) * 1e9:5.1f} ns per test row and class)  coverage {S[np.arange(2000), yy[2000:]].mean():.3f}  mean set {S.sum(1).mean():8.1f}  input {P.nbytes / 1e6:6.0f} MB, peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3:6.0f} MB")
        del P, S

if "cir" in which:
    print("CIR: coverage and mean length (standard errors), 500 calibration points, 300 repetitions of 2,000 test points")
    K = 100; lev = np.linspace(0.001, 0.999, K + 1)
    def dgp(name, rng, n):
        x = rng.uniform(0.5, 2, n)
        if name == "skewed (gamma)": e = rng.gamma(2.0, 1.0, n); base = np.quantile(rng.gamma(2.0, 1.0, 50_000), lev)
        elif name == "symmetric (normal)": e = rng.normal(size=n); base = np.quantile(rng.normal(size=50_000), lev)
        else:
            mix = lambda m: np.where(rng.uniform(size=m) < 0.5, rng.normal(-2, 0.5, m), rng.normal(2, 0.5, m)); e = mix(n); base = np.quantile(mix(50_000), lev)
        Q = np.sort(x[:, None] * base[None, :] * np.exp(0.15 * rng.normal(size=(n, 1))) + 0.05 * rng.normal(size=(n, K + 1)), axis=1)
        return Q, x * e
    for name in ("skewed (gamma)", "symmetric (normal)", "bimodal (two normals)"):
        for alpha in (0.1, 0.2):
            out = {k: [] for k in ("CQR", "CIR", "CIR, shortest window", "CIR+", "CTI")}; rng = np.random.default_rng(7)
            lo, hi = int(round(alpha / 2 * K)), int(round((1 - alpha / 2) * K))
            for _ in range(300):
                Qc, yc = dgp(name, rng, 500); Qt, yt = dgp(name, rng, 2000)
                iv = {"CQR": pc.ConformalizedQuantileRegressor(alpha=alpha).calibrate(quantiles=Qc[:, [lo, hi]], y=yc).predict_interval(quantiles=Qt[:, [lo, hi]]),
                      "CIR": pc.ConformalInterquantileRegressor(alpha=alpha).calibrate(quantiles=Qc, y=yc).predict_interval(quantiles=Qt),
                      "CIR, shortest window": pc.ConformalInterquantileRegressor(alpha=alpha, window="shortest").calibrate(quantiles=Qc, y=yc).predict_interval(quantiles=Qt),
                      "CIR+": pc.ConformalInterquantileRegressor(alpha=alpha, plus=True).calibrate(quantiles=Qc, y=yc).predict_interval(quantiles=Qt)}
                for k, v in iv.items(): out[k].append((np.mean((v[:, 0] <= yt) & (yt <= v[:, 1])), np.mean(v[:, 1] - v[:, 0])))
                cti = pc.ConformalThresholdedIntervals(alpha=alpha).calibrate(quantiles=Qc, y=yc); keep = cti.predict_set(quantiles=Qt)
                b = (Qt < yt[:, None]).sum(1); b[(b == 0) & (yt >= Qt[:, 0])] = 1   # column of the set that holds the outcome
                out["CTI"].append((np.mean(keep[np.arange(2000), b]), np.mean(cti.predict_size(quantiles=Qt))))
            line = "; ".join(f"{k} {np.mean([a for a, _ in v]):.4f} ({np.std([a for a, _ in v], ddof=1) / np.sqrt(300):.4f}) / {np.mean([w for _, w in v]):.3f} ({np.std([w for _, w in v], ddof=1) / np.sqrt(300):.3f})" for k, v in out.items())
            print(f"  {name}, target {1 - alpha:.0%}: {line}")
