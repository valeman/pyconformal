"""Level-2 'conformal bias': the value the calibration data ask for against the value applied after clipping.
Also: observed frequency of y <= predicted quantile on fresh data."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from conformal_tights import ConformalCoherentQuantileRegressor as CT
Q = (0.05, 0.25, 0.5, 0.75, 0.95); R = 6
want, got, freq, cross = [], [], [], []
for seed in range(R):
    rng = np.random.default_rng(seed)
    def data(n):
        X = rng.normal(size=(n, 5)); return X, 10 + 2 * X[:, 0] + np.sin(2 * X[:, 1]) + (0.5 + np.abs(X[:, 2])) * rng.standard_t(4, size=n)
    X, y = data(5000); Xt, yt = data(3000)
    m = CT(XGBRegressor(n_estimators=60, max_depth=3), nonconformity_estimator=XGBRegressor(n_estimators=60, max_depth=3), random_state=seed).fit(X, y)
    import time; t0 = time.time(); P = m.predict_quantiles(Xt, quantiles=Q); print(f"fit {seed}: first predict_quantiles call {time.time()-t0:.0f} s", flush=True)
    freq.append((yt[:, None] <= P).mean(0)); cross.append(np.mean(np.any(np.diff(P, axis=1) < 0, axis=1)))
    l1 = m.conformal_l1_["Δŷ"][Q]
    r = -m.residuals_calib_l2_[:, None] - l1.predict(m.ŷ_calib_l2_nonconformity_)
    want.append([np.quantile(r[:, j], q) for j, q in enumerate(Q)]); got.append(m.conformal_l2_["Δŷ"][Q])
want, got, freq = map(np.array, (want, got, freq))
print("level                         ", *[f"{q:7.2f}" for q in Q])
print("mean |shift asked by the data|", *[f"{v:7.4f}" for v in np.abs(want).mean(0)])
print("mean |shift applied|          ", *[f"{v:7.4f}" for v in np.abs(got).mean(0)])
print("share of fits where clipped   ", *[f"{v:7.2f}" for v in (np.abs(want - got) > 1e-12).mean(0)])
print("observed frequency (target = level)", *[f"{v:7.4f}" for v in freq.mean(0)])
print("share of test points with crossing quantiles:", np.mean(cross))
