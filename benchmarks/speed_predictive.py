"""Conformal predictive systems: pyconformal against crepes. Normalized system; seconds, best of 3."""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from crepes import ConformalPredictiveSystem as Ref
from pyconformal import ConformalPredictiveSystem as Ours
print("crepes", version("crepes"))
rng = np.random.default_rng(0)
def best(f, reps=3):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r
print(f"{'task':34} {'n_cal':>8} {'n_test':>9} {'crepes':>9} {'pyconformal':>12} {'speed-up':>9}")
for n, m in [(1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000)]:
    yc, pc, sc = rng.normal(size=n), np.zeros(n), rng.uniform(0.5, 3, n)
    pt, st = rng.normal(size=m), rng.uniform(0.5, 3, m); yt = pt + st * rng.normal(size=m)
    ref = Ref().fit(yc - pc, sigmas=sc); ours = Ours().fit(yc, pc, sc)
    tr, r = best(lambda: ref.predict(pt, sigmas=st, lower_percentiles=[5], higher_percentiles=[95]))
    to, o = best(lambda: ours.predict_interval(pt, 0.1, st))
    assert np.allclose(r, o)
    print(f"{'90% intervals':34} {n:8,} {m:9,} {tr:9.4f} {to:12.4f} {tr / to:8.1f}x")
    tr, r = best(lambda: ref.predict(pt, sigmas=st, y=yt, smoothing=False))
    to, o = best(lambda: ours.predict_cdf(pt, yt, st, tau=1.0))
    assert np.allclose(np.ravel(r), o)
    print(f"{'distribution function at outcomes':34} {n:8,} {m:9,} {tr:9.4f} {to:12.4f} {tr / to:8.1f}x")
for n, m in [(1_000, 2_000), (5_000, 5_000)]:
    yc, pc = rng.normal(size=n), np.zeros(n); pt = rng.normal(size=m); yt = pt + rng.normal(size=m)
    ref = Ref().fit(yc - pc); ours = Ours().fit(yc, pc)
    tr, r = best(lambda: ref.evaluate(pt, yt, metrics=["CRPS"])["CRPS"], 1)
    to, o = best(lambda: ours.crps(pt, yt).mean())
    assert abs(r - o) < 1e-9 * max(1, abs(o))
    print(f"{'CRPS (standard system)':34} {n:8,} {m:9,} {tr:9.4f} {to:12.5f} {tr / to:8.0f}x")
n, m = 100_000, 1_000_000
yc, pc = rng.normal(size=n), np.zeros(n); pt = rng.normal(size=m); yt = pt + rng.normal(size=m)
to, _ = best(lambda: Ours().fit(yc, pc).crps(pt, yt).mean())
print(f"{'CRPS (standard system)':34} {n:8,} {m:9,} {'not run':>9} {to:12.5f}")
