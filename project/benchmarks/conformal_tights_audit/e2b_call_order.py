"""Does predict_interval change what a later predict_quantiles call returns?
predict_interval(coverage=c) caches its two shifts under the key ((1-c)/2, 1-(1-c)/2) and then edits the cached
array in place (priority="coverage"). A later predict_quantiles call with exactly those two numbers reuses the edited array.
The first-pass test (e2_call_order.py) passed quantiles=(0.05, 0.95), which is a different key from
((1-0.9)/2, 1-(1-0.9)/2) = (0.04999999999999999, 0.95), so it could not detect this."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from conformal_tights import ConformalCoherentQuantileRegressor as CT
for cov, label in ((0.5, "coverage 0.5, quantiles (0.25, 0.75): keys equal"), (0.9, "coverage 0.9, quantiles (0.05, 0.95): keys differ"), (0.9, "coverage 0.9, same floating-point quantiles as predict_interval")):
    found, big = 0, 0.0
    for seed in range(20):
        rng = np.random.default_rng(seed); X = rng.normal(size=(3000, 5)); y = 10 + 2 * X[:, 0] + (0.5 + np.abs(X[:, 2])) * rng.standard_t(4, size=3000)
        m = CT(random_state=seed).fit(X, y)
        q = ((1 - cov) / 2, 1 - (1 - cov) / 2) if ("same" in label or cov == 0.5) else (0.05, 0.95)
        if cov == 0.5: q = (0.25, 0.75)
        before = np.asarray(m.predict_quantiles(X[:500], quantiles=q)).copy(); m.predict_interval(X[:500], coverage=cov)
        after = np.asarray(m.predict_quantiles(X[:500], quantiles=q)); d = np.abs(before - after).max(); found += d > 0; big = max(big, d / y.std())
    print(f"{label}: {found} of 20 fits changed; largest change {big:.4f} standard deviations of y", flush=True)
