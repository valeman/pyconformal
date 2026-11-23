"""crepes 0.9.1: p-values, online mode, wrappers with difficulty estimates and out-of-bag calibration, evaluation metrics, martingales."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from crepes import ConformalRegressor, ConformalPredictiveSystem, ConformalClassifier, WrapRegressor, WrapClassifier
from crepes.extras import DifficultyEstimator, MondrianCategorizer, binning, hinge, margin
from crepes.martingales import SimpleJumper, SleeperStayer, SleeperDrifter, CompositeMartingale, semi_online_p_values
R = int(sys.argv[1])
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
print("(e) p-values of true outcomes must be uniform: share below 0.1 and below 0.05 (smoothed)")
for label, draw in (("continuous scores", lambda g, n: g.normal(size=n)), ("heavily tied scores", lambda g, n: np.round(g.normal(size=n)))):
    for name in ("ConformalRegressor.predict_p", "ConformalPredictiveSystem.predict(y=)", "ConformalClassifier.predict_p"):
        ps = []
        for r in range(R):
            g = np.random.default_rng(r); y = draw(g, 150)
            if name.startswith("ConformalRegressor"): p = ConformalRegressor().fit(y[:100]).predict_p(np.zeros(50), y[100:])
            elif name.startswith("ConformalPredictiveSystem"): p = ConformalPredictiveSystem().fit(y[:100]).predict(np.zeros(50), y=y[100:])
            else: p = ConformalClassifier().fit(np.abs(y[:100])).predict_p(np.abs(y[100:]).reshape(-1, 1))[:, 0]
            ps.append(np.asarray(p).ravel())
        ps = np.concatenate(ps); a, b = (ps < 0.1).mean(), (ps < 0.05).mean()
        print(f"  {label:20s} {name:40s}: P(p<0.1) = {a:.4f}, P(p<0.05) = {b:.4f}, mean {ps.mean():.4f}  {'ok' if abs(a - 0.1) < 0.01 and abs(b - 0.05) < 0.008 and abs(ps.mean() - 0.5) < 0.01 else 'NOT UNIFORM'}")
print("\n(f) online mode: intervals must equal a batch fit on calibration plus all earlier test points; and cover")
for name, make in (("ConformalRegressor", ConformalRegressor), ("ConformalPredictiveSystem", ConformalPredictiveSystem)):
    for variant in ("standard", "normalized", "Mondrian"):
        diff = 0.0; cov = []
        for r in range(max(20, R // 10)):
            g = np.random.default_rng(31 * r); n, T = 60, 80; sig = np.exp(g.normal(0, 0.4, n + T)); bins = g.integers(0, 2, n + T)
            y = g.normal(size=n + T) * (sig if variant == "normalized" else 1)
            kc = dict(sigmas=sig[:n] if variant == "normalized" else None, bins=bins[:n] if variant == "Mondrian" else None)
            kt = dict(sigmas=sig[n:] if variant == "normalized" else None, bins=bins[n:] if variant == "Mondrian" else None)
            on = make().fit(y[:n], **kc).predict_int_online(np.zeros(T), y[n:], confidence=0.9, **kt)
            for t in range(0, T, 7):
                kk = dict(sigmas=sig[:n + t] if variant == "normalized" else None, bins=bins[:n + t] if variant == "Mondrian" else None)
                k1 = dict(sigmas=sig[n + t:n + t + 1] if variant == "normalized" else None, bins=bins[n + t:n + t + 1] if variant == "Mondrian" else None)
                b = make().fit(y[:n + t], **kk).predict_int(np.zeros(1), confidence=0.9, **k1)[0]
                d = np.abs(np.nan_to_num(on[t] - b, nan=0.0, posinf=0.0, neginf=0.0)).max(); diff = max(diff, d if np.isinf(on[t]).tolist() == np.isinf(b).tolist() else np.inf)
            cov.append(((on[:, 0] <= y[n:]) & (y[n:] <= on[:, 1])).mean())
        m, se = np.mean(cov), np.std(cov) / np.sqrt(len(cov))
        print(f"  {name:26s} {variant:11s}: largest difference from batch {diff:.2e}; coverage {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}")
print("\n(g) wrappers end to end with a random forest, fresh test data, 90% level")
def data(g, n):
    X = g.uniform(-2, 2, (n, 5)); y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + g.normal(size=n) * (0.2 + np.abs(X[:, 2])); return X, y
reps = max(15, R // 15)
def run(label, build):
    cov = []; wid = []; err = None
    for r in range(reps):
        g = np.random.default_rng(1000 + r); Xtr, ytr = data(g, 600); Xc, yc = data(g, 300); Xt, yt = data(g, 1000)
        try: iv = np.asarray(build(Xtr, ytr, Xc, yc, Xt, r))
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"; break
        cov.append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean()); wid.append(np.median(iv[:, 1] - iv[:, 0]))
    if err: print(f"  {label:58s} ERROR {err}"); return
    m, se = np.mean(cov), np.std(cov) / np.sqrt(reps)
    print(f"  {label:58s} coverage {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}; median width {np.mean(wid):.2f}", flush=True)
rf = lambda r: RandomForestRegressor(n_estimators=100, oob_score=True, random_state=r, n_jobs=1)
def W(est, X, y):
    w = WrapRegressor(est); w.fit(X, y); return w      # fit() returns None in crepes 0.9.1, so it cannot be chained
def std(Xtr, ytr, Xc, yc, Xt, r): return W(rf(r), Xtr, ytr).calibrate(Xc, yc).predict_int(Xt, confidence=0.9)
def cps(Xtr, ytr, Xc, yc, Xt, r): return W(rf(r), Xtr, ytr).calibrate(Xc, yc, cps=True).predict_int(Xt, confidence=0.9)
def de_knn_x(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); de = DifficultyEstimator().fit(X=Xtr, scaler=True); return w.calibrate(Xc, yc, de=de).predict_int(Xt, confidence=0.9)
def de_knn_y(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); de = DifficultyEstimator().fit(X=Xtr, y=ytr, scaler=True); return w.calibrate(Xc, yc, de=de).predict_int(Xt, confidence=0.9)
def de_knn_res_oob(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); de = DifficultyEstimator().fit(X=Xtr, residuals=ytr - w.learner.oob_prediction_, scaler=True); return w.calibrate(Xc, yc, de=de).predict_int(Xt, confidence=0.9)
def de_knn_res_insample(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); de = DifficultyEstimator().fit(X=Xtr, residuals=ytr - w.predict(Xtr), scaler=True); return w.calibrate(Xc, yc, de=de).predict_int(Xt, confidence=0.9)
def de_var(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); de = DifficultyEstimator().fit(X=Xtr, learner=w.learner, scaler=True); return w.calibrate(Xc, yc, de=de).predict_int(Xt, confidence=0.9)
def mond(Xtr, ytr, Xc, yc, Xt, r):
    w = W(rf(r), Xtr, ytr); mc = MondrianCategorizer().fit(Xc, learner=w.learner, no_bins=5); return w.calibrate(Xc, yc, mc=mc).predict_int(Xt, confidence=0.9)
def oob_plain(Xtr, ytr, Xc, yc, Xt, r):
    X = np.vstack([Xtr, Xc]); y = np.r_[ytr, yc]; return W(rf(r), X, y).calibrate(X, y, oob=True).predict_int(Xt, confidence=0.9)
def oob_de(Xtr, ytr, Xc, yc, Xt, r):
    X = np.vstack([Xtr, Xc]); y = np.r_[ytr, yc]; w = W(rf(r), X, y); de = DifficultyEstimator().fit(X=X, learner=w.learner, scaler=True, oob=True)
    return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=0.9)
def oob_de_knn(Xtr, ytr, Xc, yc, Xt, r):
    X = np.vstack([Xtr, Xc]); y = np.r_[ytr, yc]; w = W(rf(r), X, y); de = DifficultyEstimator().fit(X=X, residuals=y - w.learner.oob_prediction_, scaler=True, oob=True)
    return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=0.9)
def oob_small_forest(Xtr, ytr, Xc, yc, Xt, r):
    X = np.vstack([Xtr, Xc])[:200]; y = np.r_[ytr, yc][:200]; return W(RandomForestRegressor(n_estimators=20, oob_score=True, random_state=r), X, y).calibrate(X, y, oob=True).predict_int(Xt, confidence=0.9)
for label, f in (("split, standard", std), ("split, predictive system", cps), ("split, difficulty: k-NN distance", de_knn_x), ("split, difficulty: k-NN label spread", de_knn_y),
                 ("split, difficulty: k-NN out-of-bag residuals", de_knn_res_oob), ("split, difficulty: k-NN in-sample residuals", de_knn_res_insample), ("split, difficulty: forest variance", de_var),
                 ("split, Mondrian categorizer (5 bins)", mond), ("out-of-bag calibration, standard", oob_plain), ("out-of-bag calibration + forest-variance difficulty (oob)", oob_de),
                 ("out-of-bag calibration + k-NN residual difficulty (oob)", oob_de_knn), ("out-of-bag calibration, 200 rows, 20 trees", oob_small_forest)):
    run(label, f)
# classifier wrapper
def cdata(g, n):
    X = g.normal(size=(n, 6)); W = np.array([[2, 0, -1], [0, 2, -1], [1, 1, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0]]); v = X @ W + g.gumbel(size=(n, 3)); return X, v.argmax(1)
for label, kw, pk in (("hinge", {}, {}), ("margin", {"nc": margin}, {}), ("class-conditional", {"class_cond": True}, {}), ("out-of-bag", {"oob": True}, {}), ("hinge, no smoothing", {}, {"smoothing": False})):
    cov = []; size = []; err = None
    for r in range(reps):
        g = np.random.default_rng(2000 + r); Xtr, ytr = cdata(g, 500); Xc, yc = cdata(g, 300); Xt, yt = cdata(g, 1000)
        try:
            w = WrapClassifier(RandomForestClassifier(n_estimators=100, oob_score=True, random_state=r))
            if kw.get("oob"): X = np.vstack([Xtr, Xc]); y = np.r_[ytr, yc]; w.fit(X, y); w.calibrate(X, y, oob=True)
            else: w.fit(Xtr, ytr); w.calibrate(Xc, yc, **kw)
            S = np.asarray(w.predict_set(Xt, confidence=0.9, **pk))
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"; break
        cov.append(S[np.arange(len(yt)), yt].mean()); size.append(S.sum(1).mean())
    if err: print(f"  WrapClassifier, {label:42s} ERROR {err}"); continue
    m, se = np.mean(cov), np.std(cov) / np.sqrt(reps)
    print(f"  WrapClassifier, {label:42s} coverage {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}; mean set size {np.mean(size):.2f}", flush=True)
print("\n(h) evaluate() against hand calculations")
g = np.random.default_rng(5); n = 400; sig = np.exp(g.normal(0, 0.5, n + 300)); y = g.normal(size=n + 300) * sig; yh = np.zeros(300)
cr = ConformalRegressor().fit(y[:n], sigmas=sig[:n]); iv = cr.predict_int(yh, sigmas=sig[n:], confidence=0.9); ev = cr.evaluate(yh, y[n:], sigmas=sig[n:], confidence=0.9)
hand = {"error": 1 - ((iv[:, 0] <= y[n:]) & (y[n:] <= iv[:, 1])).mean(), "eff_mean": (iv[:, 1] - iv[:, 0]).mean(), "eff_med": np.median(iv[:, 1] - iv[:, 0])}
print("  ConformalRegressor:", {k: ("ok" if np.isclose(ev[k], v) else f"DIFFERS crepes {ev[k]:.5f} hand {v:.5f}") for k, v in hand.items()}, "| keys", sorted(ev))
for normalized in (False, True):
    cps = ConformalPredictiveSystem().fit(y[:n], sigmas=sig[:n] if normalized else None)
    kw = dict(sigmas=sig[n:]) if normalized else {}
    ev = cps.evaluate(yh, y[n:], confidence=0.9, **kw); cpds = cps.predict(yh, return_cpds=True, **kw)
    def crps_exact(c, yy):          # CRPS of the empirical distribution that puts mass 1/n on each value in c
        c = np.sort(c); return np.mean(np.abs(c - yy)) - 0.5 * np.mean(np.abs(c[:, None] - c[None, :]))
    def crps_step(c, yy):           # same, integrating (F(t) - 1{t >= y})^2 with F stepping by 1/n at each value
        c = np.sort(c); m = len(c); F = np.arange(1, m) / m; w = np.diff(c); t = 0.0
        for j in range(m - 1):
            a, b = c[j], c[j + 1]
            if yy <= a: t += (1 - F[j]) ** 2 * (b - a)
            elif yy >= b: t += F[j] ** 2 * (b - a)
            else: t += F[j] ** 2 * (yy - a) + (1 - F[j]) ** 2 * (b - yy)
        return t + max(c[0] - yy, 0) + max(yy - c[-1], 0)
    h1 = np.mean([crps_step(cpds[i], y[n + i]) for i in range(300)]); h2 = np.mean([crps_exact(cpds[i], y[n + i]) for i in range(300)])
    print(f"  ConformalPredictiveSystem, {'normalized' if normalized else 'standard  '}: CRPS crepes {ev['CRPS']:.5f}; by definition on crepes' own distributions {h1:.5f} (closed form {h2:.5f})  {'ok' if np.isclose(ev['CRPS'], h1, rtol=1e-6) else 'DIFFERS by ' + format(100 * (ev['CRPS'] / h1 - 1), '+.1f') + '%'}")
A = 1 - g.dirichlet(np.ones(4), 700); yk = g.integers(0, 4, 700); cc = ConformalClassifier().fit(A[np.arange(400), yk[:400]])
S = cc.predict_set(A[400:], confidence=0.9, seed=1); ev = cc.evaluate(A[400:], np.arange(4), yk[400:], confidence=0.9, seed=1)
hand = {"error": 1 - S[np.arange(300), yk[400:]].mean(), "avg_c": S.sum(1).mean(), "one_c": (S.sum(1) == 1).mean(), "empty": (S.sum(1) == 0).mean()}
print("  ConformalClassifier:", {k: ("ok" if np.isclose(ev[k], v) else f"DIFFERS crepes {ev[k]:.5f} hand {v:.5f}") for k, v in hand.items()}, "| keys", sorted(ev))
print("\n(i) martingales on uniform p-values: share of runs in which the martingale ever reaches 20 (must be at most 0.05)")
Rm = R * 2
for name, make in (("SimpleJumper", SimpleJumper), ("SleeperStayer", SleeperStayer), ("SleeperDrifter", SleeperDrifter), ("Composite(SimpleJumper, SleeperStayer)", lambda: CompositeMartingale([SimpleJumper(), SleeperStayer()]))):
    hit = 0; hitd = 0; err = None
    for r in range(Rm):
        g = np.random.default_rng(900 + r)
        try:
            m = np.asarray(make().apply(g.random(1000), return_index=False), dtype=float); hit += np.nanmax(m) >= 20
            p = np.r_[g.random(500), g.beta(0.5, 1.5, 500)]; hitd += np.nanmax(np.asarray(make().apply(p, return_index=False), dtype=float)) >= 20
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"; break
    if err: print(f"  {name:42s} ERROR {err}"); continue
    se = np.sqrt(0.05 * 0.95 / Rm)
    print(f"  {name:42s} uniform: {hit/Rm:.4f} {'ABOVE 5%' if hit/Rm > 0.05 + 3*se else 'ok'}; after a change at step 500: {hitd/Rm:.3f}", flush=True)
a = SimpleJumper(); g = np.random.default_rng(1); p1, p2 = g.random(300), g.random(300)
r_fresh = np.asarray(SimpleJumper().apply(p2, return_index=False), dtype=float); a.apply(p1, return_index=False); b = SimpleJumper(); r_after = np.asarray(b.apply(p2, return_index=False), dtype=float)
print(f"  a second SimpleJumper built after another one was used gives the same path as a fresh one: {bool(np.allclose(r_fresh, r_after))}")
ps = []
for r in range(R):
    g = np.random.default_rng(r); ps.append(np.asarray(semi_online_p_values(g.normal(size=200), seed=r)))
ps = np.concatenate(ps); print(f"  semi_online_p_values on i.i.d. scores: P(p<0.1) = {(ps < 0.1).mean():.4f}, mean {ps.mean():.4f}")
print("\n(j) binning()")
v = np.random.default_rng(0).normal(size=1000); b, th = binning(v, bins=10); print("  10 bins of 1,000 values: sizes", np.bincount(b).tolist())
vt = np.round(v); bt, _ = binning(vt, bins=10); print("  10 bins of heavily tied values: sizes", np.bincount(bt).tolist(), "| equal values always in the same bin:", all(len(set(bt[vt == u])) == 1 for u in np.unique(vt)))
