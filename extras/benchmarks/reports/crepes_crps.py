import numpy as np
from importlib.metadata import version
from crepes import ConformalPredictiveSystem
print("crepes", version("crepes"))

def crps_by_definition(support, y):
    """Integral of (F(t) - 1{t >= y})^2 dt for the empirical distribution on `support`."""
    pts = np.unique(np.concatenate([support, [y]]))
    left = pts[:-1]
    F = np.searchsorted(np.sort(support), left, side="right") / support.size
    return float(np.sum((F - (left >= y)) ** 2 * np.diff(pts)))

residuals = np.array([-1.0, 0.0, 1.0, 3.0])   # calibration residuals, calibration sigmas all 1
y_hat, y = 10.0, 11.0
cps = ConformalPredictiveSystem().fit(residuals, sigmas=np.ones(4))
for sigma in (1.0, 2.0, 5.0):
    cpd = cps.predict(np.array([y_hat]), sigmas=np.array([sigma]), return_cpds=True)
    got = cps.evaluate(np.array([y_hat]), np.array([y]), sigmas=np.array([sigma]), metrics=["CRPS"])["CRPS"]
    print(f"sigma={sigma}: crepes {got:.4f}, definition on crepes' own distribution {crps_by_definition(np.ravel(cpd), y):.4f}")
