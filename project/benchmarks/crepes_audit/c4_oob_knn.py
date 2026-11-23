"""crepes 0.9.1: out-of-bag calibration with the k-nearest-neighbour difficulty estimators (leave-one-out, as the DifficultyEstimator docstring describes),
and the classifier wrapper. 90% level, fresh test data."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from crepes import WrapRegressor, WrapClassifier
from crepes.extras import DifficultyEstimator, margin
reps = int(sys.argv[1])
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
def data(g, n):
    X = g.uniform(-2, 2, (n, 5)); y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + g.normal(size=n) * (0.2 + np.abs(X[:, 2])); return X, y
def W(r, X, y, trees=100):
    w = WrapRegressor(RandomForestRegressor(n_estimators=trees, oob_score=True, random_state=r)); w.fit(X, y); return w
def run(label, build, n=900, level=0.9):
    cov = []; wid = []
    for r in range(reps):
        g = np.random.default_rng(1000 + r); X, y = data(g, n); Xt, yt = data(g, 2000)
        iv = np.asarray(build(X, y, Xt, r, level)); cov.append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean()); wid.append(np.median(iv[:, 1] - iv[:, 0]))
    m, se = np.mean(cov), np.std(cov) / np.sqrt(reps)
    print(f"  {label:66s} n={n:4d} level {level}: coverage {m:.4f} (se {se:.4f}) {verdict(m, se, level)}; median width {np.mean(wid):.2f}", flush=True)
def oob_none(X, y, Xt, r, lv): w = W(r, X, y); return w.calibrate(X, y, oob=True).predict_int(Xt, confidence=lv)
def oob_var(X, y, Xt, r, lv):
    w = W(r, X, y); de = DifficultyEstimator().fit(X=X, learner=w.learner, scaler=True, oob=True); return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=lv)
def oob_knn_dist(X, y, Xt, r, lv):
    w = W(r, X, y); de = DifficultyEstimator().fit(X=X, scaler=True, oob=True); return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=lv)
def oob_knn_y(X, y, Xt, r, lv):
    w = W(r, X, y); de = DifficultyEstimator().fit(X=X, y=y, scaler=True, oob=True); return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=lv)
def oob_knn_res(X, y, Xt, r, lv):
    w = W(r, X, y); de = DifficultyEstimator().fit(X=X, residuals=y - w.learner.oob_prediction_, scaler=True, oob=True); return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=lv)
def oob_knn_res_k5(X, y, Xt, r, lv):
    w = W(r, X, y); de = DifficultyEstimator().fit(X=X, residuals=y - w.learner.oob_prediction_, scaler=True, oob=True, k=5); return w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=lv)
def split_knn_res(X, y, Xt, r, lv):
    h = len(X) * 2 // 3; w = W(r, X[:h], y[:h]); de = DifficultyEstimator().fit(X=X[:h], residuals=y[:h] - w.learner.oob_prediction_, scaler=True); return w.calibrate(X[h:], y[h:], de=de).predict_int(Xt, confidence=lv)
print("Regression")
for n in (900, 300):
    for label, f in (("out-of-bag, no difficulty estimate", oob_none), ("out-of-bag + forest variance (oob=True)", oob_var), ("out-of-bag + k-NN distance (oob=True)", oob_knn_dist),
                     ("out-of-bag + k-NN label spread (oob=True)", oob_knn_y), ("out-of-bag + k-NN residuals (oob=True), k=25", oob_knn_res), ("out-of-bag + k-NN residuals (oob=True), k=5", oob_knn_res_k5),
                     ("split (2/3 train, 1/3 calibration) + k-NN residuals", split_knn_res)):
        run(label, f, n)
run("out-of-bag + k-NN residuals (oob=True), k=25", oob_knn_res, 900, 0.95); run("out-of-bag + k-NN residuals (oob=True), k=25", oob_knn_res, 900, 0.8)
print("Classification")
def cdata(g, n):
    X = g.normal(size=(n, 6)); Wm = np.array([[2, 0, -1], [0, 2, -1], [1, 1, 0], [0, 0, 2], [0, 0, 0], [0, 0, 0]]); v = X @ Wm + g.gumbel(size=(n, 3)); return X, v.argmax(1)
for label, kw, pk in (("hinge", {}, {}), ("margin", {"nc": margin}, {}), ("class-conditional", {"class_cond": True}, {}), ("out-of-bag", {"oob": True}, {}), ("hinge, no smoothing", {}, {"smoothing": False})):
    cov = []; size = []; cc = [[], [], []]
    for r in range(reps):
        g = np.random.default_rng(2000 + r); Xtr, ytr = cdata(g, 500); Xc, yc = cdata(g, 300); Xt, yt = cdata(g, 1500)
        w = WrapClassifier(RandomForestClassifier(n_estimators=100, oob_score=True, random_state=r))
        if kw.get("oob"): X = np.vstack([Xtr, Xc]); y = np.r_[ytr, yc]; w.fit(X, y); w.calibrate(X, y, oob=True)
        else: w.fit(Xtr, ytr); w.calibrate(Xc, yc, **kw)
        S = np.asarray(w.predict_set(Xt, confidence=0.9, labels=False, **pk)); hit = S[np.arange(len(yt)), yt]
        cov.append(hit.mean()); size.append(S.sum(1).mean())
        for c in range(3): cc[c].append(hit[yt == c].mean())
    m, se = np.mean(cov), np.std(cov) / np.sqrt(reps)
    print(f"  WrapClassifier, {label:22s}: coverage {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}; mean set size {np.mean(size):.2f}; per class {[round(float(np.mean(c)), 3) for c in cc]}", flush=True)
