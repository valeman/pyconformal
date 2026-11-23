import numpy as np
from mapie.risk_control.methods import get_r_hat_plus

delta, n, p, R = 0.1, 100, 0.1, 2000
grid = np.arange(0, 1, 0.01)                     # MAPIE's default grid of thresholds
below = 0
for r in range(R):
    x = (np.random.default_rng(r).random(n) < p).astype(float)      # 0/1 losses with true risk 0.1
    risks = np.repeat(x[:, None], len(grid), axis=1)
    upper = get_r_hat_plus(risks, grid, "rcps", None, delta, 0.25)[1][0]   # default bound (WSR)
    below += upper < p
print(f"upper confidence bound below the true risk in {below / R:.3f} of runs; allowed {delta}")
