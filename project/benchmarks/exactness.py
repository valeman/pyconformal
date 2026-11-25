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

IMPORTS = dict(pyconformal="pyconformal", mapie="mapie", crepes="crepes", puncc="deel.puncc",
               conformalpy="conformalpy", conformalforecast="conformalforecast")
# Exceptions that are a deliberate refusal of a level the calibration set cannot support. Anything else is an error.
REFUSALS = ("Insufficient", "ValueError")

rng = np.random.default_rng(0)
alphas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.9]
ns = list(range(1, 121)) + [199, 200, 999, 1000]
cases = [(n, a, rng.exponential(size=n)) for n in ns for a in alphas]
n_inf = sum(math.isinf(exact(s, a)) for _, a, s in cases)
print(f"{len(cases)} cases (n = 1..120, 199, 200, 999, 1000; alphas {alphas}); "
      f"in {n_inf} of them no finite interval is valid\n")
cols = ("exact finite", "exact infinite", "refused infeasible", "too narrow", "too wide", "refused feasible", "error")
print(f"{'package':18} " + " ".join(f"{c:>18}" for c in cols) + "  example")
for name, f in PKGS.items():
    try:
        __import__(IMPORTS[name])
    except Exception as x:                                 # a package that is not installed scores nothing
        print(f"{name:18} not installed ({type(x).__name__}); skipped"); continue
    c = dict.fromkeys(cols, 0); ex = ""
    for n, a, s in cases:
        e = exact(s, a)
        try:
            h = f(s, a)
        except Exception as x:
            msg = type(x).__name__
            if any(t in msg for t in REFUSALS):
                c["refused infeasible" if math.isinf(e) else "refused feasible"] += 1
                if not math.isinf(e): ex = ex or f"refused feasible n={n},a={a}: {msg}"
            else:
                c["error"] += 1; ex = ex or f"n={n},a={a}: {msg}: {str(x)[:60]}"
            continue
        if h is None or (isinstance(h, float) and math.isnan(h)):
            c["error"] += 1; ex = ex or f"n={n},a={a}: returned {h}"; continue
        if math.isinf(e) and h == math.inf: c["exact infinite"] += 1          # the upper end must be +inf, not -inf
        elif not math.isinf(e) and abs(h - e) <= 1e-12 * max(1, abs(e)): c["exact finite"] += 1
        elif h < e: c["too narrow"] += 1; ex = ex or f"n={n},a={a}: got {h:.4g}, exact {e:.4g}"
        else: c["too wide"] += 1; ex = ex or f"n={n},a={a}: got {h:.4g}, exact {e:.4g}"
    print(f"{name:18} " + " ".join(f"{c[k]:18d}" for k in cols) + f"  {ex}")
