"""(1) Cause: is MAPIE's WSR bound exactly 'run the product separately on two halves and add the two maxima'?
(2) Faster grid of settings for the bound itself.
(3) End to end through the public MultiLabelClassificationController with method='rcps'."""
import sys, numpy as np, warnings
from mapie.risk_control.methods import get_r_hat_plus
from mapie.risk_control import MultiLabelClassificationController
warnings.simplefilter("ignore")
grid = np.arange(0, 1, 0.01)   # MAPIE's own default grid of thresholds

def wsr(x, delta, grid, split):
    n = len(x); i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (i + 1)
    s2 = np.r_[0.25, ((0.25 + np.cumsum((x - mu) ** 2)) / (i + 1))[:-1]]
    nu = np.minimum(1, np.sqrt(2 * np.log(1 / delta) / (n * s2)))
    logs = np.log(1 - nu[:, None] * (x[:, None] - grid[None, :]) + np.finfo(float).eps)
    if split:   # what the code in mapie/risk_control/methods.py does
        h = n // 2
        K = np.cumsum(logs[:h], axis=0).max(axis=0) + np.cumsum(logs[n - h:], axis=0).max(axis=0)
    else:       # Bates et al. (2021): one running product over all n observations
        K = np.cumsum(logs, axis=0).max(axis=0)
    ok = np.where(K > np.log(1 / delta))[0]
    return grid[ok[0]] if len(ok) else 1.0

R = int(sys.argv[1])
same = tot = 0
print("Share of runs in which the upper bound falls BELOW the true risk (must be at most delta)")
for kind in ("binary", "beta"):
    for p in (0.1, 0.2, 0.3):
        for n in (50, 100, 200, 300):
            for delta in (0.1, 0.05):
                bad = [0, 0, 0]
                for r in range(R):
                    rng = np.random.default_rng(104729 * r + n)
                    x = (rng.random(n) < p).astype(float) if kind == "binary" else rng.beta(2 * p / (1 - p), 2.0, n)
                    um = get_r_hat_plus(np.repeat(x[:, None], len(grid), axis=1), grid, "rcps", None, delta, 0.25)[1][0]
                    us, up = wsr(x, delta, grid, True), wsr(x, delta, grid, False)
                    same += um == us; tot += 1
                    bad[0] += um < p; bad[1] += us < p; bad[2] += up < p
                se = np.sqrt(delta * (1 - delta) / R)
                print(f"  {kind:6s} true risk {p:4} n={n:4d} delta={delta:4}: MAPIE default bound {bad[0]/R:.3f}   two-halves replica {bad[1]/R:.3f}   paper's bound {bad[2]/R:.3f}  {'ABOVE delta' if bad[0]/R > delta + 3*se else 'ok'}", flush=True)
print(f"MAPIE's bound equals the two-halves replica in {same} of {tot} runs")

# (3) end to end
K = 6
def ml(n, rng):
    P = rng.beta(1, 2, (n, K)); Y = (rng.random((n, K)) < P).astype(int); keep = Y.sum(1) > 0
    return P[keep], Y[keep]
rng = np.random.default_rng(1); Pb, Yb = ml(400000, rng)
lam = np.arange(0, 1, 0.01)
true_risk = np.array([1 - (((Pb > l) & (Yb == 1)).sum(1) / Yb.sum(1)).mean() for l in lam])   # 1 - recall at each threshold
R2 = int(sys.argv[2])
print("\nEnd to end, MultiLabelClassificationController(risk='recall', method='rcps'): share of runs whose TRUE risk exceeds the target (must be at most 1 - confidence_level)")
for target, conf, ncal in ((0.8, 0.9, 110), (0.7, 0.9, 110), (0.8, 0.9, 330)):
    for bound in (None, "hoeffding"):
        bad = 0
        for r in range(R2):
            rng = np.random.default_rng(7919 * r + ncal); Pc, Yc = ml(ncal, rng)
            c = MultiLabelClassificationController(lambda X: np.asarray(X), risk="recall", method="rcps", target_level=target, confidence_level=conf, rcps_bound=bound)
            c.calibrate(Pc, Yc)
            l = float(np.ravel(c.best_predict_param)[0])
            bad += true_risk[int(round(l * 100))] > 1 - target
        d = 1 - conf; se = np.sqrt(d * (1 - d) / R2)
        print(f"  recall >= {target}, confidence {conf}, about {int(ncal * 0.91)} calibration rows, rcps_bound={str(bound):9s}: {bad/R2:.3f} (allowed {d:.2f})  {'VIOLATED' if bad/R2 > d + 3*se else 'ok'}", flush=True)
