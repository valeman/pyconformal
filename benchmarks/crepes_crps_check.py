"""crepes: CRPS of a normalized conformal predictive system against the definition.
CRPS(F, y) = integral of (F(t) - 1{t >= y})^2 dt, with F the predictive distribution."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from crepes import ConformalPredictiveSystem
from pyconformal import ConformalPredictiveSystem as Ours
print("crepes", version("crepes"))
def exact(support, y):
    """Exact integral of the squared step function, segment by segment."""
    pts = np.unique(np.concatenate([support, [y]]))
    left = pts[:-1]
    F = np.searchsorted(np.sort(support), left, side="right") / support.size
    return float(np.sum((F - (left >= y)) ** 2 * np.diff(pts)))
resid = np.array([-1.0, 0.0, 1.0, 3.0]); y_hat, y = 10.0, 11.0
print("One test point: prediction 10, outcome 11, calibration scores", resid.tolist())
print(f"{'sigma':>6} {'definition':>11} {'crepes':>8} {'pyconformal':>12}")
for sigma in (1.0, 2.0, 5.0, 0.2):
    cps = ConformalPredictiveSystem().fit(resid, sigmas=np.ones(4))
    c = cps.evaluate(np.array([y_hat]), np.array([y]), sigmas=np.array([sigma]), metrics=["CRPS"])["CRPS"]
    o = Ours().fit(resid, np.zeros(4), np.ones(4)).crps([y_hat], [y], [sigma])[0]
    print(f"{sigma:6.1f} {exact(y_hat + sigma * resid, y):11.4f} {c:8.4f} {o:12.4f}")
rng = np.random.default_rng(0); n, m = 500, 2000
yc, pc, sc = rng.normal(size=n), np.zeros(n), rng.uniform(0.5, 3, n)
pt, st = np.zeros(m), rng.uniform(0.5, 3, m); yt = st * rng.normal(size=m)
c = ConformalPredictiveSystem().fit(yc - pc, sigmas=sc).evaluate(pt, yt, sigmas=st, metrics=["CRPS"])["CRPS"]
ours = Ours().fit(yc, pc, sc); o = ours.crps(pt, yt, st).mean()
C = ours.predict_distribution(pt[:200], st[:200])
print(f"\nRandom normalized example, mean CRPS over {m} test points: crepes {c:.4f}, pyconformal {o:.4f}")
print("pyconformal against the definition on the first 200:", np.allclose([exact(C[i], yt[i]) for i in range(200)], ours.crps(pt[:200], yt[:200], st[:200]), atol=1e-6))
