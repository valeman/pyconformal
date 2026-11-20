"""Prediction sets on the same probabilities: pyconformal, TorchCP, MAPIE, PUNCC.
Coverage, mean set size and time (calibrate + predict, best of 3). Target 90%."""
import time, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np, torch
from importlib.metadata import version
from sklearn.base import BaseEstimator, ClassifierMixin
print("torchcp", version("torchcp"), "| mapie", version("mapie"), "| puncc", version("puncc"))
class Pass(BaseEstimator, ClassifierMixin):
    def __init__(self, K=2): self.K = K
    def fit(self, X, y): self.classes_ = np.arange(self.K); return self
    def predict_proba(self, X): return np.asarray(X)
    def predict(self, X): return np.asarray(X).argmax(1)
def data(rng, n, K):
    y = rng.integers(0, K, size=n); lg = rng.normal(size=(n, K)) * 1.5 + 4.0 * np.eye(K)[y]
    P = np.exp(lg - lg.max(1, keepdims=True)); return P / P.sum(1, keepdims=True), y
def best(f, reps=3):
    b, r = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r
def ours(score):
    from pyconformal import SplitConformalClassifier
    def f():
        m = SplitConformalClassifier(est, 0.1, prefit=True, score=score, random_state=0).calibrate(Pc, yc); return m.predict_set(Pt)
    return f
def tcp(mk):
    from torchcp.classification.predictor import SplitPredictor
    lc, lt, ycT = torch.tensor(np.log(Pc + 1e-300)), torch.tensor(np.log(Pt + 1e-300)), torch.tensor(yc)
    def f():
        torch.manual_seed(0); p = SplitPredictor(mk(), model=None, alpha=0.1); p.calculate_threshold(lc, ycT)
        s = p.predict_with_logits(lt); return (s if torch.is_tensor(s) else torch.tensor(np.asarray(s))).numpy().astype(bool)
    return f
def mapie(score):
    from mapie.classification import SplitConformalClassifier as M
    def f():
        m = M(est, confidence_level=0.9, conformity_score=score, prefit=True, random_state=0); m.conformalize(Pc, yc); return m.predict_set(Pt)[1][:, :, 0]
    return f
def puncc(cls, **kw):
    from deel.puncc.api.prediction import BasePredictor
    class Proba(BasePredictor):
        def predict(self, X, **k): return self.model.predict_proba(X)
    def f():
        cp = cls(Proba(est, is_trained=True), train=False, **kw); cp.fit(X_calib=Pc, y_calib=yc); _, sets = cp.predict(Pt, alpha=0.1)
        out = np.zeros(Pt.shape, dtype=bool)
        for i, s in enumerate(sets): out[i, list(s)] = True
        return out
    return f
from torchcp.classification.score import APS as TA, RAPS as TR, SAPS as TS, LAC as TL
from deel.puncc.classification import APS as PA, RAPS as PR, LAC as PL
from pyconformal import RAPS, SAPS, APS
for K, n_cal, n_test in [(10, 5_000, 50_000), (100, 5_000, 50_000), (1000, 5_000, 20_000)]:
    rng = np.random.default_rng(K); Pc, yc = data(rng, n_cal, K); Pt, yt = data(rng, n_test, K); est = Pass(K).fit(None, None)
    print(f"\n{K} classes, {n_cal:,} calibration and {n_test:,} test points")
    print(f"{'method':22} {'library':12} {'coverage':>9} {'set size':>9} {'seconds':>9}")
    rows = [("LAC", "pyconformal", ours("lac")), ("LAC", "TorchCP", tcp(lambda: TL())), ("LAC", "MAPIE", mapie("lac")), ("LAC", "PUNCC", puncc(PL)),
            ("APS randomised", "pyconformal", ours("aps")), ("APS randomised", "TorchCP", tcp(lambda: TA())), ("APS randomised", "PUNCC", puncc(PA)),
            ("APS not randomised", "pyconformal", ours(APS(randomized=False))), ("APS (default)", "MAPIE", mapie("aps")),
            ("RAPS randomised", "pyconformal", ours(RAPS(0.01, 5))), ("RAPS randomised", "TorchCP", tcp(lambda: TR(penalty=0.01, kreg=5))),
            ("RAPS (default)", "MAPIE", mapie("raps")), ("RAPS", "PUNCC", puncc(PR, lambd=0.01, k_reg=5)),
            ("SAPS randomised", "pyconformal", ours(SAPS(0.2))), ("SAPS randomised", "TorchCP", tcp(lambda: TS(weight=0.2))),
            ("SAPS not randomised", "pyconformal", ours(SAPS(0.2, randomized=False))), ("SAPS not randomised", "TorchCP", tcp(lambda: TS(weight=0.2, randomized=False)))]
    for name, lib, f in rows:
        try:
            t, s = best(f, 1 if lib in ("PUNCC", "MAPIE") and K >= 100 else 3)
            print(f"{name:22} {lib:12} {s[np.arange(n_test), yt].mean():9.4f} {s.sum(1).mean():9.2f} {t:9.3f}", flush=True)
        except Exception as e:
            print(f"{name:22} {lib:12} ERR {type(e).__name__}: {str(e)[:70]}", flush=True)
