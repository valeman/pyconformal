"""Calibrators on fresh test data, with a deliberately over-confident model (Gaussian naive Bayes on correlated features)."""
import numpy as np, warnings
warnings.simplefilter("ignore")
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from mapie.calibration import TopLabelCalibrator, VennAbersCalibrator
from mapie.metrics.calibration import top_label_ece
def tece(y, P, B=15):   # top-label ECE by hand, equal-width bins
    conf = np.nanmax(P, 1); pred = np.nanargmax(P, 1); corr = (pred == y).astype(float); e = 0.0
    idx = np.clip((conf * B).astype(int), 0, B - 1)
    for b in range(B):
        m = idx == b
        if m.any(): e += m.sum() * abs(corr[m].mean() - conf[m].mean())
    return e / len(y)
def show(label, f):
    try: print(f"{label:66s} {f()}", flush=True)
    except Exception as e: print(f"{label:66s} ERROR {type(e).__name__}: {' '.join(str(e).split())[:170]}", flush=True)
for K in (2, 4):
    X, y = make_classification(n_samples=9000, n_features=20, n_informative=8, n_redundant=8, n_classes=K, n_clusters_per_class=2, random_state=K)
    Xtr, ytr, Xc, yc, Xt, yt = X[:3000], y[:3000], X[3000:6000], y[3000:6000], X[6000:], y[6000:]
    base = GaussianNB().fit(Xtr, ytr); P0 = base.predict_proba(Xt)
    print(f"\n{K} classes. Raw model: top-label ECE {tece(yt, P0):.4f}, log loss {log_loss(yt, P0):.4f}, accuracy {(P0.argmax(1) == yt).mean():.4f}")
    for cal in ("sigmoid", "isotonic"):
        def tl():
            m = TopLabelCalibrator(estimator=base, calibrator=cal, cv="prefit").fit(Xc, yc); P = np.asarray(m.predict_proba(Xt))
            nanrows = np.isnan(P).any(1).mean(); s = np.nansum(P, 1)
            return f"top-label ECE {tece(yt, P):.4f}; predicted class unchanged {(np.nanargmax(P, 1) == P0.argmax(1)).mean():.3f}; rows containing NaN {nanrows:.2f}; row sums {s.min():.3f} to {s.max():.3f}; predict() accuracy {(m.predict(Xt) == yt).mean():.4f}"
        show(f"TopLabelCalibrator, {cal}, prefit", tl)
        def tls():
            m = TopLabelCalibrator(estimator=GaussianNB(), calibrator=cal, cv="split").fit(X[:6000], y[:6000]); P = np.asarray(m.predict_proba(Xt))
            return f"top-label ECE {tece(yt, P):.4f}"
        show(f"TopLabelCalibrator, {cal}, split", tls)
    def va(**kw):
        def f():
            m = VennAbersCalibrator(estimator=kw.pop("est", GaussianNB()), random_state=0, **kw)
            m.fit(*( (Xc, yc) if kw.get("cv") == "prefit" else (X[:6000], y[:6000]) )); P = np.asarray(m.predict_proba(Xt)); s = P.sum(1)
            return f"top-label ECE {tece(yt, P):.4f}; log loss {log_loss(yt, np.clip(P, 1e-12, 1)):.4f}; row sums {s.min():.4f} to {s.max():.4f}; values in [0,1] {bool(((P >= 0) & (P <= 1)).all())}; accuracy {(P.argmax(1) == yt).mean():.4f}"
        return f
    show("VennAbersCalibrator, prefit", va(est=base, cv="prefit"))
    show("VennAbersCalibrator, inductive (default)", va(inductive=True))
    show("VennAbersCalibrator, cross, 5 splits", va(inductive=False, n_splits=5))
    # against the reference package, same calibration data
    def ref():
        from venn_abers import VennAbersCalibrator as R
        r = R(estimator=base, inductive=True, cal_size=None) if False else None
        m = VennAbersCalibrator(estimator=base, cv="prefit").fit(Xc, yc); Pm = np.asarray(m.predict_proba(Xt))
        from venn_abers import VennAbersCalibrator as RV
        rv = RV(); Pr = rv.predict_proba(p_cal=base.predict_proba(Xc), y_cal=yc, p_test=base.predict_proba(Xt))
        return f"max difference from the venn-abers package {np.abs(Pm - np.asarray(Pr)).max():.2e}"
    show("VennAbersCalibrator, prefit, against venn-abers package", ref)
