"""Input handling: the same problem given as NumPy, pandas, sparse, with string labels, several levels, weights, groups, one test row."""
import numpy as np, pandas as pd, scipy.sparse as sp, warnings
warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from mapie.regression import SplitConformalRegressor, CrossConformalRegressor, JackknifeAfterBootstrapRegressor, ConformalizedQuantileRegressor
from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
rng = np.random.default_rng(0)
n = 1200; X = rng.normal(size=(n, 4)); y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=n)
Xtr, ytr, Xc, yc, Xt, yt = X[:500], y[:500], X[500:900], y[500:900], X[900:], y[900:]
def run(label, f):
    try: print(f"{label:74s} {f()}")
    except Exception as e: print(f"{label:74s} ERROR {type(e).__name__}: {str(e)[:150]}")
def reg(Xtr, ytr, Xc, yc, Xt, **kw):
    m = SplitConformalRegressor(LinearRegression(), confidence_level=0.9, prefit=False, **kw).fit(Xtr, ytr).conformalize(Xc, yc); return m.predict_interval(Xt)[1]
base = reg(Xtr, ytr, Xc, yc, Xt)
same = lambda a: "same intervals as NumPy" if np.allclose(np.asarray(a), base) else f"DIFFERENT from NumPy (max gap {np.abs(np.asarray(a) - base).max():.4f})"
cols = list("abcd")
run("regression: pandas DataFrame and Series", lambda: same(reg(pd.DataFrame(Xtr, columns=cols), pd.Series(ytr), pd.DataFrame(Xc, columns=cols), pd.Series(yc), pd.DataFrame(Xt, columns=cols))))
run("regression: pandas with a shuffled, non-default index", lambda: same(reg(pd.DataFrame(Xtr, columns=cols, index=rng.permutation(500) + 7), pd.Series(ytr, index=rng.permutation(500) + 99), pd.DataFrame(Xc, columns=cols, index=rng.permutation(400)), pd.Series(yc, index=rng.permutation(400) + 5), pd.DataFrame(Xt, columns=cols))))
run("regression: SciPy sparse X", lambda: same(reg(sp.csr_matrix(Xtr), ytr, sp.csr_matrix(Xc), yc, sp.csr_matrix(Xt))))
run("regression: Python lists", lambda: same(reg(Xtr.tolist(), ytr.tolist(), Xc.tolist(), yc.tolist(), Xt.tolist())))
run("regression: y as a column vector (n, 1)", lambda: same(reg(Xtr, ytr.reshape(-1, 1), Xc, yc.reshape(-1, 1), Xt)))
run("regression: float32 inputs", lambda: f"max gap to float64 {np.abs(reg(Xtr.astype('f4'), ytr.astype('f4'), Xc.astype('f4'), yc.astype('f4'), Xt.astype('f4')) - base).max():.2e}")
run("regression: one test row", lambda: same(np.vstack([reg(Xtr, ytr, Xc, yc, Xt[i:i+1]) for i in range(len(Xt))])))
run("regression: NaN in calibration y", lambda: "accepted; interval finite: " + str(np.isfinite(reg(Xtr, ytr, Xc, np.where(np.arange(400) == 3, np.nan, yc), Xt)).all()))
run("regression: NaN in test X", lambda: "accepted" if reg(Xtr, ytr, Xc, yc, np.where(np.arange(1200).reshape(300, 4) == 5, np.nan, Xt)) is not None else "")
def levels():
    m = SplitConformalRegressor(LinearRegression(), confidence_level=[0.8, 0.9, 0.95], prefit=False).fit(Xtr, ytr).conformalize(Xc, yc); iv = m.predict_interval(Xt)[1]
    w = iv[:, 1, :] - iv[:, 0, :]; one = [SplitConformalRegressor(LinearRegression(), confidence_level=c, prefit=False).fit(Xtr, ytr).conformalize(Xc, yc).predict_interval(Xt)[1][:, :, 0] for c in (0.8, 0.9, 0.95)]
    return f"shape {iv.shape}; widths nested: {bool((np.diff(w, axis=1) >= 0).all())}; each level equals its single-level run: {all(np.allclose(iv[:, :, i], one[i]) for i in range(3))}"
run("regression: three levels at once", levels)
run("regression: levels given unsorted [0.95, 0.8, 0.9]", lambda: (lambda iv: f"column order follows input: {bool(np.allclose(iv[:, :, 1], SplitConformalRegressor(LinearRegression(), confidence_level=0.8, prefit=False).fit(Xtr, ytr).conformalize(Xc, yc).predict_interval(Xt)[1][:, :, 0]))}")(SplitConformalRegressor(LinearRegression(), confidence_level=[0.95, 0.8, 0.9], prefit=False).fit(Xtr, ytr).conformalize(Xc, yc).predict_interval(Xt)[1]))
# weights: a weight of zero must act like removing the row from the fit
def weights():
    w = np.ones(900); w[:200] = 0
    a = CrossConformalRegressor(Ridge(), confidence_level=0.9, cv=5, random_state=0).fit_conformalize(X[:900], y[:900], fit_params={"sample_weight": w}).predict_interval(Xt)[1]
    b = CrossConformalRegressor(Ridge(), confidence_level=0.9, cv=5, random_state=0).fit_conformalize(X[:900], y[:900]).predict_interval(Xt)[1]
    return f"sample_weight changes the intervals: {not np.allclose(a, b)} (mean width {np.mean(a[:,1]-a[:,0]):.3f} against {np.mean(b[:,1]-b[:,0]):.3f} unweighted)"
run("cross-conformal: sample_weight through fit_params", weights)
def groups():
    g = np.repeat(np.arange(90), 10); m = CrossConformalRegressor(LinearRegression(), confidence_level=0.9, cv=GroupKFold(5)).fit_conformalize(X[:900], y[:900], groups=g); iv = m.predict_interval(Xt)[1][:, :, 0]
    return f"runs; coverage on test {((iv[:,0] <= yt) & (yt <= iv[:,1])).mean():.3f}"
run("cross-conformal: GroupKFold with groups", groups)
run("jackknife+-after-bootstrap: pandas input", lambda: (lambda iv: f"runs; coverage {((iv[:,0] <= yt) & (yt <= iv[:,1])).mean():.3f}")(JackknifeAfterBootstrapRegressor(LinearRegression(), confidence_level=0.9, random_state=0).fit_conformalize(pd.DataFrame(X[:900], columns=cols), pd.Series(y[:900])).predict_interval(pd.DataFrame(Xt, columns=cols))[1][:, :, 0]))
# classification
K = 4; Xk = rng.normal(size=(1500, 5)); W = rng.normal(size=(5, K)) * 1.5; yk = np.array([rng.choice(K, p=np.exp(v - v.max()) / np.exp(v - v.max()).sum()) for v in Xk @ W])
def clf(ylab, score="lac", Xf=lambda a: a):
    m = SplitConformalClassifier(LogisticRegression(max_iter=500), confidence_level=0.9, conformity_score=score, prefit=False, random_state=0).fit(Xf(Xk[:600]), ylab[:600]).conformalize(Xf(Xk[600:1100]), ylab[600:1100])
    return np.asarray(m.predict_set(Xf(Xk[1100:]))[1])[:, :, 0]
bc = clf(yk)
for score in ("lac", "aps"):
    b = clf(yk, score)
    run(f"classification ({score}): string labels", lambda: "same sets as integer labels" if np.array_equal(clf(np.array(list("wxyz"))[yk], score), b) else "DIFFERENT sets from integer labels")
    run(f"classification ({score}): labels 10, 20, 30, 40", lambda: "same sets" if np.array_equal(clf((yk + 1) * 10, score), b) else "DIFFERENT sets")
    run(f"classification ({score}): labels in reverse numeric meaning (3 - y)", lambda: "same sets (columns reversed)" if np.array_equal(clf(3 - yk, score)[:, ::-1], b) else "DIFFERENT sets")
    run(f"classification ({score}): pandas input", lambda: "same sets" if np.array_equal(clf(pd.Series(yk), score, lambda a: pd.DataFrame(a)), b) else "DIFFERENT sets")
    run(f"classification ({score}): a class missing from the calibration set", lambda: (lambda yy: "runs" if clf(yy, score) is not None else "")(np.where((np.arange(1500) >= 600) & (np.arange(1500) < 1100) & (yk == 3), 0, yk)))
run("classification (lac): boolean labels, two classes", lambda: "runs; set shape " + str(np.asarray(SplitConformalClassifier(LogisticRegression(), confidence_level=0.9, prefit=False).fit(Xk[:600], yk[:600] > 1).conformalize(Xk[600:1100], yk[600:1100] > 1).predict_set(Xk[1100:])[1]).shape))
