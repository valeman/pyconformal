"""CQR, CTI and CIR on precomputed quantile predictions.

CQR: pyconformal against MAPIE and PUNCC (same symmetric correction).
CTI: pyconformal against the authors' code (github.com/luo-lorry/CTI); pass the
path of a clone as the first argument to include it.
Calibrate + predict, best of 5, seconds.
"""
import sys, time, types, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
from importlib.metadata import version
import pyconformal as pc

def best(f, reps=5):
    b, r = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r
rng = np.random.default_rng(0)
def data(n, K):
    x = rng.uniform(0.5, 2, n); y = x * rng.gamma(2.0, 1.0, n)
    base = np.quantile(rng.gamma(2.0, 1.0, 50_000), np.linspace(0.001, 0.999, K + 1))
    return np.sort(x[:, None] * base[None, :] * np.exp(0.15 * rng.normal(size=(n, 1))) + 0.05 * rng.normal(size=(n, K + 1)), axis=1), y

class Col:  # a "model" that returns one stored column
    def __init__(self, j): self.j = j
    def fit(self, X, y): return self
    def predict(self, X): return X[:, self.j]
    def __sklearn_is_fitted__(self): return True

print("CQR: 100,000 calibration and 1,000,000 test points")
Qc, yc = data(100_000, 2); Qt, yt = data(1_000_000, 2); Qc, Qt = Qc[:, [0, 2]], Qt[:, [0, 2]]
def ours(): return pc.ConformalizedQuantileRegressor(alpha=0.1).calibrate(quantiles=Qc, y=yc).predict_interval(quantiles=Qt)
t0, r0 = best(ours); cov = lambda iv: ((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean()
print(f"  pyconformal {t0 * 1e3:9.2f} ms   coverage {cov(r0):.4f}")
try:
    from mapie.regression import ConformalizedQuantileRegressor as M
    from sklearn.base import BaseEstimator, RegressorMixin
    class SK(BaseEstimator, RegressorMixin):
        def __init__(self, j=0): self.j = j
        def fit(self, X, y): self.f_ = True; return self
        def predict(self, X): return X[:, self.j]
    Q3c, Q3t = np.column_stack([Qc, Qc.mean(1)]), np.column_stack([Qt, Qt.mean(1)])
    def mapie():
        m = M([SK(0).fit(0, 0), SK(1).fit(0, 0), SK(2).fit(0, 0)], confidence_level=0.9, prefit=True)
        m.conformalize(Q3c, yc); return m.predict_interval(Q3t, symmetric_correction=True)[1][:, :, 0]
    t, r = best(mapie, 3); print(f"  MAPIE {version('mapie'):8}{t * 1e3:8.2f} ms   coverage {cov(r):.4f}   {t / t0:.0f}x slower   same intervals: {np.allclose(r, r0)}")
except Exception as e: print("  MAPIE not run:", type(e).__name__, str(e)[:80])
try:
    from deel.puncc.regression import CQR
    from deel.puncc.api.prediction import DualPredictor
    def puncc():
        cp = CQR(DualPredictor([Col(0), Col(1)], is_trained=[True, True]), train=False)
        cp.fit(X_calib=Qc, y_calib=yc); _, lo, hi = cp.predict(Qt, alpha=0.1); return np.column_stack([lo, hi])
    t, r = best(puncc, 3); print(f"  PUNCC {version('puncc'):8}{t * 1e3:8.2f} ms   coverage {cov(r):.4f}   {t / t0:.0f}x slower   same intervals: {np.allclose(r, r0)}")
except Exception as e: print("  PUNCC not run:", type(e).__name__, str(e)[:80])

print("\nInterquantile methods: 100 intervals, 20,000 calibration and 200,000 test points")
Qc, yc = data(20_000, 100); Qt, yt = data(200_000, 100)
def cti(): return pc.ConformalThresholdedIntervals().calibrate(quantiles=Qc, y=yc).predict_set(quantiles=Qt)
t_cti, s_cti = best(cti)
b = (Qt < yt[:, None]).sum(1); b[(b == 0) & (yt >= Qt[:, 0])] = 1
print(f"  CTI  pyconformal {t_cti * 1e3:8.1f} ms   coverage {s_cti[np.arange(len(yt)), b].mean():.4f}")
if len(sys.argv) > 1:
    src = open(sys.argv[1].rstrip("/") + "/torchcp/regression/predictors/cti.py").read()
    src = src.replace("from torchcp.regression.utils.metrics import Metrics", "Metrics = object").replace("from torchcp.utils.common import calculate_conformal_value", "")
    ref = types.ModuleType("cti_ref"); exec(src, ref.__dict__)
    obj = ref.CTI(None, quantiles=[0.5])
    def authors():
        pad = lambda Q: np.hstack([np.full((len(Q), 1), -np.inf), Q, np.full((len(Q), 1), np.inf)])
        lengths = obj.find_containing_intervals(pad(Qc), yc)[1]
        n = len(lengths); q = np.sort(lengths)[int(np.ceil((n + 1) * 0.9)) - 1]
        P = pad(Qt); mask = np.diff(P, axis=1) <= q
        return np.stack((np.where(mask, P[:, :-1], np.nan), np.where(mask, P[:, 1:], np.nan)), axis=2), mask
    t, (r, mask) = best(authors, 3)
    print(f"  CTI  authors' code {t * 1e3:6.1f} ms   {t / t_cti:.1f}x slower   same sets: {np.array_equal(mask, s_cti)}")
for name, plus in (("CIR ", False), ("CIR+", True)):
    def cir(): return pc.ConformalInterquantileRegressor(plus=plus).calibrate(quantiles=Qc, y=yc).predict_interval(quantiles=Qt)
    t, iv = best(cir)
    print(f"  {name} pyconformal {t * 1e3:8.1f} ms   coverage {((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean():.4f}   mean width {(iv[:, 1] - iv[:, 0]).mean():.3f}")
cq = pc.ConformalizedQuantileRegressor().calibrate(quantiles=Qc[:, [5, 95]], y=yc).predict_interval(quantiles=Qt[:, [5, 95]])
print(f"  CQR on the same data: coverage {((cq[:, 0] <= yt) & (yt <= cq[:, 1])).mean():.4f}   mean width {(cq[:, 1] - cq[:, 0]).mean():.3f}")
print(f"  CTI mean set size {pc.ConformalThresholdedIntervals().calibrate(quantiles=Qc, y=yc).predict_size(quantiles=Qt).mean():.3f}")
