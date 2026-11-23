"""NOTE: in part (a) the direct call to _compute_regression_quantile below passes the wrong arguments
(reverse=True with alpha in place of 1 - alpha), so its "too low" count is an error of this script, not of MAPIE.
The correct call is in t8b.py, where the function is exact in 37,581 of 37,581 cases. The classification part and part (b) are right.

Checks that follow from reading mapie/utils.py and mapie/conformity_scores/regression.py (1.6.0).
(a) The two shared quantile functions, called directly, against the exact rank, over a wide grid.
(b) Width minimisation picks beta on the calibration data itself: does coverage survive the selection?"""
import math, sys, numpy as np
from fractions import Fraction
from common import *
from mapie.utils import _compute_regression_quantile, _compute_classification_quantile
from mapie.regression import SplitConformalRegressor, ConformalizedQuantileRegressor
from mapie.conformity_scores import AbsoluteConformityScore, GammaConformityScore

# (a) exact rank? levels as a user types them; MAPIE's public classes pass alpha = 1 - confidence_level
levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999]
out = {"reg": [0, 0, 0, 0], "clf": [0, 0, 0, 0]}   # exact, one too high, too low, refused/other although finite exists
ex = {"reg": [], "clf": []}
for n in range(1, 3001):
    s = np.arange(1.0, n + 1)                       # the k-th smallest score is k
    for lv in levels:
        k = math.ceil((n + 1) * Fraction(repr(lv)))
        if k > n: continue
        a = np.array([1 - lv])
        for name in ("reg", "clf"):
            try:
                if name == "reg": q = float(_compute_regression_quantile(s.reshape(-1, 1), a, axis=0, reverse=True)[0, 0])
                else: q = float(_compute_classification_quantile(s.reshape(-1, 1), a)[0])
            except Exception:
                out[name][3] += 1; ex[name].append((n, lv, "refused")) if len(ex[name]) < 12 else None; continue
            i = 0 if q == k else (1 if q == k + 1 else (2 if q < k else 3))
            out[name][i] += 1
            if i and len(ex[name]) < 12: ex[name].append((n, lv, q, k))
for name, label in (("reg", "_compute_regression_quantile"), ("clf", "_compute_classification_quantile")):
    e, h, l, r = out[name]
    print(f"{label}: {e + h + l + r} cases with a finite exact rank: exact {e}, one rank too high {h}, too low {l}, refused or other {r}")
    print("   examples (n, level, got, exact rank):", ex[name][:8])

# (b) coverage with and without width minimisation, skewed errors, 2,000 repetitions
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
print("\nWidth minimisation (beta chosen on the calibration scores): coverage at a 90% target, skewed errors")
for label, make, shift in (("split regression, absolute score, sym=False", lambda: AbsoluteConformityScore(sym=False), 0.0),
                           ("split regression, gamma score", lambda: GammaConformityScore(), 50.0)):
    for n_cal in (30, 50, 100, 300):
        for miw in (False, True):
            cov, wid, err = [], [], None
            for r in range(R):
                rng = np.random.default_rng(7919 * r + n_cal)
                y = shift + rng.gamma(1.0, 1.0, n_cal + 300) - 1.0   # skewed, mean zero around the prediction
                X = np.zeros((n_cal + 300, 1))
                try:
                    m = SplitConformalRegressor(Const(shift), confidence_level=0.9, conformity_score=make(), prefit=True).conformalize(X[:n_cal], y[:n_cal])
                    iv = m.predict_interval(X[n_cal:], minimize_interval_width=miw)[1][:, :, 0]
                    cov.append(((iv[:, 0] <= y[n_cal:]) & (y[n_cal:] <= iv[:, 1])).mean()); wid.append(np.mean(iv[:, 1] - iv[:, 0]))
                except Exception as e:
                    err = f"{type(e).__name__}: {str(e)[:60]}"; break
            if err: print(f"  {label:44s} n_cal={n_cal:4d} minimise={str(miw):5s} REFUSES ({err})"); continue
            m_, se = summary(cov)
            print(f"  {label:44s} n_cal={n_cal:4d} minimise={str(miw):5s} coverage {m_:.4f} (se {se:.4f}) width {np.mean(wid):.3f}  {verdict(m_, se, 0.9)}", flush=True)
