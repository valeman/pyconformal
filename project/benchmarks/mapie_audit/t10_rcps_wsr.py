"""RCPS upper confidence bounds, called directly: the bound must fall below the true risk in at most delta of runs.
Also the same data through a WSR bound written from Bates et al. (2021), eq. for the Waudby-Smith-Ramdas bound (one running product over all n)."""
import sys, numpy as np, warnings
from mapie.risk_control.methods import get_r_hat_plus
warnings.simplefilter("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
grid = np.linspace(0, 1, 201)                      # MAPIE uses the lambda grid as its grid of candidate risk values

def wsr_paper(x, delta, grid):
    n = len(x); i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(x)) / (i + 1)
    s2 = (0.25 + np.cumsum((x - mu) ** 2)) / (i + 1)
    s2 = np.r_[0.25, s2[:-1]]
    nu = np.minimum(1, np.sqrt(2 * np.log(1 / delta) / (n * s2)))
    K = np.cumsum(np.log(1 - nu[:, None] * (x[:, None] - grid[None, :]) + np.finfo(float).eps), axis=0).max(axis=0)
    ok = np.where(K > np.log(1 / delta))[0]
    return grid[ok[0]] if len(ok) else 1.0

def draw(kind, p, n, rng):
    if kind == "binary": return (rng.random(n) < p).astype(float)
    if kind == "beta": return rng.beta(2 * p / (1 - p), 2.0, n)          # mean p, values in [0, 1]
for kind in ("binary", "beta"):
    for p in (0.1, 0.3):
        for n in (100, 101, 500, 2000):
            for delta in (0.1, 0.05):
                bad = {"hoeffding": 0, "bernstein": 0, "wsr": 0, "wsr_paper": 0}; gap = []
                for r in range(R):
                    rng = np.random.default_rng(104729 * r + n)
                    x = draw(kind, p, n, rng)
                    risks = np.repeat(x[:, None], len(grid), axis=1)
                    for b in ("hoeffding", "bernstein", "wsr"):
                        u = get_r_hat_plus(risks, grid, "rcps", b, delta, 0.25)[1][0]
                        bad[b] += u < p
                        if b == "wsr": um = u
                    up = wsr_paper(x, delta, grid); bad["wsr_paper"] += up < p; gap.append(up - um)
                se = np.sqrt(delta * (1 - delta) / R)
                flag = "  <-- WSR ABOVE delta" if bad["wsr"] / R > delta + 3 * se else ""
                print(f"{kind:6s} risk {p} n={n:4d} delta={delta}: bound below the true risk in  hoeffding {bad['hoeffding']/R:.3f}  bernstein {bad['bernstein']/R:.3f}  MAPIE wsr {bad['wsr']/R:.3f}  paper wsr {bad['wsr_paper']/R:.3f}   paper minus MAPIE bound: mean {np.mean(gap):+.4f}, max {np.max(gap):+.4f}{flag}", flush=True)
