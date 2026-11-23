"""crepes 0.9.1: online classification, percentiles, pandas inputs, chaining."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from crepes import ConformalClassifier, ConformalPredictiveSystem, ConformalRegressor, WrapRegressor, WrapClassifier
from crepes.extras import DifficultyEstimator, MondrianCategorizer
K = 4; R = 300
print("online classification, 90%: coverage of the true label")
for variant in ("standard", "Mondrian", "no warm start"):
    cov = []
    for r in range(R):
        g = np.random.default_rng(r); N = 260; P = g.dirichlet(np.ones(K), N); y = np.array([g.choice(K, p=p) for p in P]); A = 1 - P; bins = g.integers(0, 2, N); n = 60
        cc = ConformalClassifier().fit(A[np.arange(n), y[:n]], bins=bins[:n] if variant == "Mondrian" else None)
        S = np.asarray(cc.predict_set_online(A[n:], np.arange(K), y[n:], bins=bins[n:] if variant == "Mondrian" else None, confidence=0.9, warm_start=variant != "no warm start"))
        cov.append(S[np.arange(N - n), y[n:]].mean())
    print(f"  {variant:14s}: {np.mean(cov):.4f} (se {np.std(cov)/np.sqrt(R):.4f})")
print("online p-values (regression), true outcomes: uniform?")
ps = []
for r in range(R):
    g = np.random.default_rng(r); y = g.normal(size=200); ps.append(np.asarray(ConformalRegressor().fit(y[:50]).predict_p_online(np.zeros(150), y[50:])).ravel())
ps = np.concatenate(ps); print(f"  P(p<0.1) = {(ps < 0.1).mean():.4f}, P(p<0.05) = {(ps < 0.05).mean():.4f}, mean {ps.mean():.4f}")
print("percentiles of the predictive system: share of outcomes below the stated percentile")
for lo, hi in ((5, 95), (10, 90), (25, 75), (50, 50)):
    a = []; b = []
    for r in range(R):
        g = np.random.default_rng(r); y = g.standard_t(4, size=1200); q = ConformalPredictiveSystem().fit(y[:200]).predict(np.zeros(1000), lower_percentiles=lo, higher_percentiles=hi)
        a.append((y[200:] < q[:, 0]).mean()); b.append((y[200:] <= q[:, 1]).mean())
    print(f"  lower {lo:2d}: {np.mean(a):.4f} (at most {lo/100}); higher {hi:2d}: {np.mean(b):.4f} (at least {hi/100})  {'ok' if np.mean(a) <= lo/100 + 0.003 and np.mean(b) >= hi/100 - 0.003 else 'CHECK'}")
print("pandas inputs and chaining")
g = np.random.default_rng(0); X = pd.DataFrame(g.normal(size=(600, 4)), columns=list("abcd"), index=g.permutation(600) + 1000); y = pd.Series(X["a"].values + g.normal(size=600), index=X.index)
def t(label, f):
    try: print(f"  {label:66s} {f()}")
    except Exception as e: print(f"  {label:66s} ERROR {type(e).__name__}: {' '.join(str(e).split())[:130]}")
w = WrapRegressor(RandomForestRegressor(n_estimators=30, oob_score=True, random_state=0))
t("WrapRegressor.fit returns", lambda: type(w.fit(X[:300], y[:300])).__name__)
wn = WrapRegressor(RandomForestRegressor(n_estimators=30, oob_score=True, random_state=0)); wn.fit(X[:300].values, y[:300].values); wn.calibrate(X[300:500].values, y[300:500].values)
t("pandas frame with shuffled index: same intervals as NumPy", lambda: bool(np.allclose(w.calibrate(X[300:500], y[300:500]).predict_int(X[500:], confidence=0.9), wn.predict_int(X[500:].values, confidence=0.9))))
t("DifficultyEstimator on a DataFrame (k-NN)", lambda: DifficultyEstimator().fit(X=X[:300], y=y[:300], scaler=True).apply(X[500:]).shape)
t("MondrianCategorizer(learner=fitted model) then apply", lambda: MondrianCategorizer().fit(X[300:500].values, learner=wn.learner, no_bins=5).apply(X[500:].values).shape)
t("calibrate with y as a list", lambda: w.calibrate(X[300:500], list(y[300:500])).predict_int(X[500:]).shape)
c = WrapClassifier(RandomForestClassifier(n_estimators=30, random_state=0)); yc = pd.Series(np.where(y > 0, "pos", "neg"), index=y.index)
t("WrapClassifier with string labels in a Series", lambda: (c.fit(X[:300], yc[:300]), c.calibrate(X[300:500], yc[300:500]), c.predict_set(X[500:505], confidence=0.9))[2])
