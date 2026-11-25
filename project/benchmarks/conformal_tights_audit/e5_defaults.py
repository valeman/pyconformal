"""As e1_coverage.py, but with conformal-tights' own defaults ("auto": XGBoost defaults, 100 trees of depth 6)."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from conformal_tights import ConformalCoherentQuantileRegressor as CT
def data(n, rng):
    X = rng.normal(size=(n, 5))
    return X, 10 + 2 * X[:, 0] + np.sin(2 * X[:, 1]) + (0.5 + np.abs(X[:, 2])) * rng.standard_t(4, size=n)
def trial(n, cov, seed):
    rng = np.random.default_rng(seed); X, y = data(n, rng); Xt, yt = data(3000, rng)
    m = CT(random_state=seed).fit(X, y); iv = m.predict_interval(Xt, coverage=cov); q = ((1 - cov) / 2, 1 - (1 - cov) / 2)
    yh = m.base_estimator_.predict(Xt); Z = m.nonconformity_estimator_.predict(Xt)
    a = yh[:, None] + m.conformal_l1_["Δŷ"][q].predict(Z) + m.conformal_l2_["Δŷ"][q]
    r = yh[:, None] + np.abs(yh[:, None]) * (m.conformal_l1_["Δŷ/ŷ"][q].predict(Z) + m.conformal_l2_["Δŷ/ŷ"][q])
    c = lambda I: np.mean((I[:, 0] <= yt) & (yt <= I[:, 1]))
    return c(iv), c(a), c(r), np.mean(iv[:, 1] - iv[:, 0]), len(m.ŷ_calib_l2_)
R = int(sys.argv[1])
print("n      target  coverage  se      abs-only  rel-only  width   L2 size  trials below target")
for n in (300, 1000, 1500, 5000):
    for cov in (0.9, 0.95):
        o = np.array([trial(n, cov, 1000 * n + s) for s in range(R)])
        print(f"{n:<6d} {cov:<7} {o[:,0].mean():.4f}    {o[:,0].std()/np.sqrt(R):.4f}  {o[:,1].mean():.4f}    {o[:,2].mean():.4f}    {o[:,3].mean():.3f}   {int(o[0,4]):<8d} {np.mean(o[:,0] < cov):.2f}", flush=True)
