"""predict_quantiles(priority='accuracy') changes after predict_interval is called with the same quantiles."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from conformal_tights import ConformalCoherentQuantileRegressor as CT
found = 0
for seed in range(20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(3000, 5)); y = 10 + 2 * X[:, 0] + (0.5 + np.abs(X[:, 2])) * rng.standard_t(4, size=3000)
    m = CT(XGBRegressor(n_estimators=60, max_depth=3), nonconformity_estimator=XGBRegressor(n_estimators=60, max_depth=3), random_state=seed).fit(X, y)
    q = (0.05, 0.95)
    before = m.predict_quantiles(X[:500], quantiles=q).copy()
    m.predict_interval(X[:500], coverage=0.9)
    after = m.predict_quantiles(X[:500], quantiles=q)
    d = np.abs(before - after).max()
    found += d > 0
    if d > 0: print(f"seed {seed}: same call, same data, largest change {d:.4f}")
print(f"{found} of 20 fits affected")
