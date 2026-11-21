"""CSP speed: pyconformal against csp-forecaster 0.2.0 (mode="fast"). fit + predict, best of 20, milliseconds."""
import time, sys, os
import numpy as np
from importlib.metadata import version
from csp_forecaster import ConformalSeasonalPool as Ref
import csp_forecaster.core as core
from pyconformal import ConformalSeasonalPool as Ours
print("csp-forecaster", version("csp-forecaster"), "| its numba kernel active:", core._USE_NUMBA)
rng = np.random.default_rng(0)
def best(f, reps=20):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
    return b * 1e3
Ref(random_state=0).fit(rng.normal(size=50), 7).predict(3)  # warm up / compile
print(f"{'history':>8} {'season':>7} {'horizon':>8} {'samples':>8} {'csp-forecaster ms':>18} {'pyconformal ms':>15} {'speed-up':>9}")
for T, m, H, n in [(200, 12, 12, 100), (5000, 24, 48, 100), (5000, 1, 30, 100), (5000, 168, 336, 100),
                   (5000, 1440, 1440, 100), (5000, 24, 48, 1000), (50_000, 24, 720, 100)]:
    y = rng.normal(size=T).cumsum()
    tr = best(lambda: Ref(random_state=0).fit(y, m).predict(H, n_samples=n), 20 if H < 1000 else 5)
    to = best(lambda: Ours(random_state=0).fit(y, m).predict(H, n_samples=n), 20 if H < 1000 else 5)
    print(f"{T:8,} {m:7,} {H:8,} {n:8,} {tr:18.3f} {to:15.3f} {tr / to:8.1f}x")
