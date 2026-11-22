"""Is the threshold the exact conformal order statistic? Split regression (absolute score) and LAC classification.
The definition: the ceil((n+1)(1-alpha))-th smallest calibration score; no finite threshold if that rank exceeds n."""
import math, numpy as np
from fractions import Fraction
from common import *
from mapie.regression import SplitConformalRegressor
from mapie.classification import SplitConformalClassifier

levels = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
res = {"exact": 0, "one_too_high": 0, "too_low": 0, "refused_but_finite_exists": 0, "refused_correctly": 0, "finite_where_none_valid": 0, "infinite_correctly": 0, "other": 0}
bad = []
for allow in (False, True):
    for n in list(range(1, 61)) + [99, 100, 199, 200, 499, 999, 1000]:
        rng = np.random.default_rng(n); s = np.sort(rng.exponential(size=n))
        X = np.zeros((n, 1)); y = s.copy()  # model predicts 0, so |y - 0| = s
        for lv in levels:
            k = math.ceil((n + 1) * Fraction(repr(lv)))
            want = math.inf if k > n else s[k - 1]
            try:
                m = SplitConformalRegressor(Const(0.0), confidence_level=lv, prefit=True).conformalize(X, y)
                iv = m.predict_interval(np.zeros((1, 1)), allow_infinite_bounds=allow)[1][0, :, 0]
                got = iv[1]
            except Exception as e:
                res["refused_correctly" if math.isinf(want) else "refused_but_finite_exists"] += 1
                if not math.isinf(want): bad.append(("refused", allow, n, lv))
                continue
            if math.isinf(want):
                if math.isinf(got): res["infinite_correctly"] += 1
                else: res["finite_where_none_valid"] += 1; bad.append(("finite", allow, n, lv, got))
            elif np.isclose(got, want, rtol=1e-12): res["exact"] += 1
            elif k < n and np.isclose(got, s[k], rtol=1e-12): res["one_too_high"] += 1; bad.append(("high", allow, n, lv))
            elif got < want: res["too_low"] += 1; bad.append(("LOW", allow, n, lv, got, want))
            else: res["other"] += 1; bad.append(("other", allow, n, lv, got, want))
print("Split regression, absolute score:", res)
print("  examples:", bad[:12])

# LAC: the set contains label j iff 1 - p_j <= q. Probe q with a fine grid of test probabilities.
res = {"exact": 0, "one_too_high": 0, "too_low": 0, "refused": 0, "full_set_correctly": 0, "finite_where_none_valid": 0, "other": 0}; bad = []
grid = np.linspace(0, 1, 20001)
Pt = np.column_stack([grid, 1 - grid])
for n in list(range(1, 61)) + [99, 100, 199, 200, 999]:
    rng = np.random.default_rng(n + 7); p_true = rng.uniform(0.02, 0.98, n)   # prob of the true label (class 0)
    P = np.column_stack([p_true, 1 - p_true]); y = np.zeros(n, dtype=int); s = np.sort(1 - p_true)
    for lv in levels:
        k = math.ceil((n + 1) * Fraction(repr(lv))); want = math.inf if k > n else s[k - 1]
        try:
            m = SplitConformalClassifier(FixedClf(2).fit(P), confidence_level=lv, conformity_score="lac", prefit=True).conformalize(P, y)
            S = m.predict_set(Pt)[1][:, 0, 0]
        except Exception as e:
            res["refused"] += 1; continue
        inc = grid[S]                      # p values for which class 0 is included
        got = 1 - inc.min() if inc.size else -1.0   # largest score still included
        if math.isinf(want):
            if S.all(): res["full_set_correctly"] += 1
            else: res["finite_where_none_valid"] += 1; bad.append(("finite", n, lv, got))
        # the probe finds the threshold from below, to within one grid step
        elif 0 <= want - got <= 1.01 / 20000: res["exact"] += 1
        elif k < n and 0 <= s[k] - got <= 1.01 / 20000: res["one_too_high"] += 1; bad.append(("high", n, lv))
        elif got < want: res["too_low"] += 1; bad.append(("LOW", n, lv, got, want))
        else: res["other"] += 1; bad.append(("other", n, lv, got, want))
print("LAC classification:", res)
print("  examples:", bad[:12])
