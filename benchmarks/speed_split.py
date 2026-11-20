import time, warnings, sys
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
from exactness import Zero
rng = np.random.default_rng(1)
N_CAL, N_TEST = 200_000, 1_000_000
s = rng.exponential(size=N_CAL); Xc = np.zeros((N_CAL, 1)); Xt = np.zeros((N_TEST, 1))
def t(f, reps=15):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best
def pyc():
    from pyconformal import SplitConformalRegressor
    m = SplitConformalRegressor(Zero().fit(None, None), alpha=0.1).calibrate(Xc, s); m.predict_interval(Xt)
def mapie():
    from mapie.regression import SplitConformalRegressor
    m = SplitConformalRegressor(Zero().fit(None, None), confidence_level=0.9, prefit=True); m.conformalize(Xc, s); m.predict_interval(Xt)
def crepes():
    from crepes import ConformalRegressor
    ConformalRegressor().fit(s).predict_int(np.zeros(N_TEST), confidence=0.9)
def puncc():
    from deel.puncc.regression import SplitCP; from deel.puncc.api.prediction import BasePredictor
    cp = SplitCP(BasePredictor(Zero().fit(None, None), is_trained=True), train=False); cp.fit(X_calib=Xc, y_calib=s); cp.predict(Xt, alpha=0.1)
def cpy():
    from conformalpy.regressor import ConformalRegressor
    m = ConformalRegressor(Zero().fit(None, None), alpha=0.1); m.calibrate(Xc, s); m.predict_interval(Xt)
print(f"split conformal regression, {N_CAL:,} calibration and {N_TEST:,} test points, best of 15 (milliseconds)")
for name, f in [("pyconformal", pyc), ("mapie", mapie), ("crepes", crepes), ("puncc", puncc), ("conformalpy", cpy)]:
    try: print(f"{name:14} {1e3*t(f):8.2f} ms")
    except Exception as e: print(name, "ERR", type(e).__name__, str(e)[:80])
