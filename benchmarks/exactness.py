"""Compare each package's split-conformal regression half-width with the exact
finite-sample order statistic: k = ceil((n+1)(1-alpha)); inf if k > n."""
import math, warnings, sys
from fractions import Fraction
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, RegressorMixin

class Zero(BaseEstimator, RegressorMixin):
    def fit(self, X, y): self.fitted_ = True; return self
    def predict(self, X): return np.zeros(len(X))

def exact(scores, alpha):
    n = len(scores); k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
    return math.inf if k > n else float(np.sort(scores)[k - 1])

def hw_pyconformal(s, a):
    from pyconformal import SplitConformalRegressor
    m = SplitConformalRegressor(alpha=a).calibrate(y_pred=np.zeros(len(s)), y=s)
    return float(m.predict_interval(y_pred=np.zeros(1))[0, 1])

def hw_mapie(s, a):
    from mapie.regression import SplitConformalRegressor
    m = SplitConformalRegressor(Zero().fit(None, None), confidence_level=1 - a, prefit=True)
    m.conformalize(np.zeros((len(s), 1)), s)
    _, iv = m.predict_interval(np.zeros((1, 1)), allow_infinite_bounds=True)
    return float(iv[0, 1, 0])

def hw_crepes(s, a):
    from crepes import ConformalRegressor
    cr = ConformalRegressor().fit(s)  # residuals
    return float(cr.predict_int(np.zeros(1), confidence=1 - a)[0, 1])

def hw_puncc(s, a):
    from deel.puncc.regression import SplitCP
    from deel.puncc.api.prediction import BasePredictor
    cp = SplitCP(BasePredictor(Zero().fit(None, None), is_trained=True), train=False)
    cp.fit(X_calib=np.zeros((len(s), 1)), y_calib=s)
    _, lo, hi = cp.predict(np.zeros((1, 1)), alpha=a)
    return float(np.asarray(hi).ravel()[0])

def hw_conformalpy(s, a):
    from conformalpy.regressor import ConformalRegressor
    m = ConformalRegressor(Zero().fit(None, None), alpha=a)
    m.calibrate(np.zeros((len(s), 1)), s)
    return float(np.asarray(m.predict_interval(np.zeros((1, 1))))[0, 1])

def hw_conformalforecast(s, a):
    from conformalforecast._quantiles import conformal_quantile
    return float(conformal_quantile(s.reshape(1, -1, 1), a)[0, 0])

PKGS = dict(pyconformal=hw_pyconformal, mapie=hw_mapie, crepes=hw_crepes,
            puncc=hw_puncc, conformalpy=hw_conformalpy, conformalforecast=hw_conformalforecast)

rng = np.random.default_rng(0)
alphas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.9]
ns = list(range(1, 121)) + [199, 200, 999, 1000]
cases = [(n, a, rng.exponential(size=n)) for n in ns for a in alphas]
print(f"{len(cases)} cases (n = 1..120, 199, 200, 999, 1000; alphas {alphas})\n")
print(f"{'package':18} {'exact':>6} {'too narrow':>10} {'too wide':>9} {'refused':>8} {'error':>6}  example")
for name, f in PKGS.items():
    ok = narrow = wide = refused = err = 0; ex = ""
    for n, a, s in cases:
        e = exact(s, a)
        try:
            h = f(s, a)
        except Exception as x:
            msg = type(x).__name__
            if math.isinf(e): ok += 1  # refusing an infeasible level is fine
            elif any(t in msg for t in ("Insufficient", "Value")): refused += 1; ex = ex or f"refused feasible n={n},a={a}: {msg}"
            else: err += 1; ex = ex or f"n={n},a={a}: {msg}: {str(x)[:60]}"
            continue
        if h is None or (isinstance(h, float) and math.isnan(h)):
            err += 1; continue
        if (math.isinf(e) and math.isinf(h)) or (not math.isinf(e) and abs(h - e) <= 1e-12 * max(1, abs(e))): ok += 1
        elif h < e: narrow += 1; ex = ex or f"n={n},a={a}: got {h:.4g}, exact {e:.4g}"
        else: wide += 1; ex = ex or f"n={n},a={a}: got {h:.4g}, exact {e:.4g}"
    print(f"{name:18} {ok:6} {narrow:10} {wide:9} {refused:8} {err:6}  {ex}")
