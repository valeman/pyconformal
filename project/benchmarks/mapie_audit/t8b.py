import math, numpy as np, warnings
from fractions import Fraction
from common import *
from mapie.utils import _compute_regression_quantile
from mapie.regression import SplitConformalRegressor
from mapie.conformity_scores import AbsoluteConformityScore
warnings.simplefilter("ignore")
# (a) regression quantile, called the way mapie/conformity_scores/regression.py line 480 calls it
levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
c = [0, 0, 0]; ex = []
for n in range(1, 3001):
    s = np.arange(1.0, n + 1)[:, None]
    for lv in levels:
        k = math.ceil((n + 1) * Fraction(repr(lv)))
        if k > n: continue
        q = float(_compute_regression_quantile(s, 1 - np.array([1 - lv]), axis=0)[0, 0])   # alpha_ref = 1 - alpha, as MAPIE forms it
        i = 0 if q == k else (1 if q > k else 2); c[i] += 1
        if i and len(ex) < 10: ex.append((n, lv, q, k))
print(f"regression quantile as MAPIE calls it: exact {c[0]}, too high {c[1]}, too low {c[2]}; examples (n, level, got, exact): {ex}")
# (b) allow_infinite_bounds with the default symmetric score and too few calibration points
rng = np.random.default_rng(0); X = np.zeros((305, 1)); y = rng.normal(size=305)
for sym in (True, False):
    m = SplitConformalRegressor(Const(0.0), confidence_level=0.9, conformity_score=AbsoluteConformityScore(sym=sym), prefit=True).conformalize(X[:5], y[:5])
    for flag in (False, True):
        try:
            iv = m.predict_interval(X[5:6], allow_infinite_bounds=flag)[1][0, :, 0]; r = f"interval {iv}"
        except Exception as e: r = f"{type(e).__name__}: {str(e)[:90]}"
        print(f"  5 calibration points, 90% level, sym={sym}, allow_infinite_bounds={flag}: {r}")
