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
