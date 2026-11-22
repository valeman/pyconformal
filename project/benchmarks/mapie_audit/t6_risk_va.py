"""Risk control and Venn-Abers on MAPIE 1.6.0."""
import warnings, numpy as np, time
from common import *
from mapie.risk_control import BinaryClassificationController, MultiLabelClassificationController
from mapie.calibration import VennAbersCalibrator

# 1. Learn-then-test: precision >= 0.6 with confidence 0.9. Across fresh calibration sets,
#    the share of runs whose TRUE precision is below 0.6 must be at most 0.1.
rng = np.random.default_rng(0)
def draw(n):
    p = rng.beta(2, 2, n); return p.reshape(-1, 1), (rng.random(n) < p).astype(int)
big_p = rng.beta(2, 2, 400000)
def true_precision(thr):   # P(y=1 | p >= thr), exactly computable from a huge sample of p
    m = big_p >= thr; return big_p[m].mean() if m.any() else 1.0
fails, none, R = 0, 0, 400
for r in range(R):
    Xc, yc = draw(500)
    try:
        c = BinaryClassificationController(lambda X: np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]]), "precision", target_level=0.6, confidence_level=0.9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); c.calibrate(Xc, yc)
        thr = c.best_predict_param
        if thr is None or (isinstance(thr, float) and np.isnan(thr)): none += 1; continue
        fails += true_precision(float(np.ravel(thr)[0])) < 0.6
    except Exception as e:
        print("LTT ERROR", type(e).__name__, str(e)[:150]); break
print(f"Learn-then-test, precision >= 0.6 at confidence 0.9: runs with true precision below 0.6: {fails}/{R} = {fails/R:.3f} (allowed 0.10); no valid threshold found in {none} runs")

# 2. Conformal risk control for multi-label recall: E[recall] >= 0.9
K = 6
def ml(n):
    P = rng.beta(1, 2, (n, K)); Y = (rng.random((n, K)) < P).astype(int)
    keep = Y.sum(1) > 0; return P[keep], Y[keep]   # the method needs at least one label per row
rec = []
for r in range(300):
    Pc, Yc = ml(330); Pt, Yt = ml(2200)
    try:
        c = MultiLabelClassificationController(lambda X: np.asarray(X), risk="recall", method="crc", target_level=0.9)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); c.calibrate(Pc, Yc); S = np.asarray(c.predict(Pt))
        S = S[:, :, 0] if S.ndim == 3 else S
        has = Yt.sum(1) > 0
        rec.append(((S & (Yt == 1)).sum(1)[has] / Yt.sum(1)[has]).mean())
    except Exception as e:
        print("CRC ERROR", type(e).__name__, str(e)[:150]); break
if rec:
    m, se = summary(rec); print(f"CRC multi-label recall, target 0.9: mean recall {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}")

# 3. Venn-Abers: warnings, and agreement with the reference package
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=6000, n_features=10, random_state=0)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    va = VennAbersCalibrator(GaussianNB().fit(X[:2000], y[:2000]), cv="prefit")
    va.fit(X[2000:4000], y[2000:4000]); p = va.predict_proba(X[4000:])
rt = [x for x in w if issubclass(x.category, RuntimeWarning)]
print(f"Venn-Abers binary, prefit: {len(rt)} RuntimeWarnings during fit and predict ({set(str(x.message)[:45] for x in rt)})")
from venn_abers import VennAbers
ref = VennAbers(); pc_, pt_ = GaussianNB().fit(X[:2000], y[:2000]).predict_proba(X[2000:4000]), GaussianNB().fit(X[:2000], y[:2000]).predict_proba(X[4000:])
ref.fit(pc_, y[2000:4000]); pr, _ = ref.predict_proba(pt_)
print(f"  max difference from the venn-abers package: {np.nanmax(np.abs(p - pr)):.2e}; NaN in output: {np.isnan(p).sum()}")
