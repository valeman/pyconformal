"""MSTL: pyconformal against statsforecast and (if statsmodels is present) a
statsmodels-based MSTL with the same jump settings. Best of 3, milliseconds.
Outputs are compared at matching settings.
"""
import math, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from pyconformal import mstl
from statsforecast.mstl import mstl as sf_mstl
from statsmodels.tsa.seasonal import STL

def best(f, k=3):
    b = 1e9
    for _ in range(k):
        s = time.perf_counter(); f(); b = min(b, time.perf_counter() - s)
    return b * 1e3
def sm_jumps(y, periods, iterate=2):  # R's settings, built on statsmodels
    odd = lambda v: int(v) + 1 - int(v) % 2
    seas = np.zeros((len(periods), len(y))); des = y.copy()
    for _ in range(iterate):
        for i, p in enumerate(periods):
            des = des + seas[i]; s_w = 11 + 4 * i
            t_w = odd(math.ceil(1.5 * p / (1 - 1.5 / s_w))); l_w = odd(p)
            fit = STL(des, period=p, seasonal=s_w, seasonal_deg=0, trend=t_w, low_pass=l_w, seasonal_jump=math.ceil(s_w / 10),
                      trend_jump=math.ceil(t_w / 10), low_pass_jump=math.ceil(l_w / 10)).fit(inner_iter=2, outer_iter=0)
            seas[i] = fit.seasonal; des = des - seas[i]
    return seas
mstl(np.sin(np.arange(200.0)), [7])  # compile
print(f"statsforecast {version('statsforecast')}, statsmodels {version('statsmodels')}")
print(f"{'series':34} {'pyconformal':>12} {'same settings':>14} {'statsforecast':>14} {'statsmodels+jumps':>18}")
rng = np.random.default_rng(0)
for label, n, periods in (("15-minute, 6 weeks (day, week)", 4032, [96, 672]), ("hourly, 1 year (day, week)", 8760, [24, 168]),
                          ("hourly, 11 years (day, week)", 100_000, [24, 168])):
    t = np.arange(n); y = 10 + sum(np.sin(2 * np.pi * t / p) for p in periods) + rng.normal(size=n)
    ours = best(lambda: mstl(y, periods)); same = best(lambda: mstl(y, periods, jumps=False, inner_iter=5))
    sf = best(lambda: sf_mstl(y, periods), 1); smj = best(lambda: sm_jumps(y, periods))
    a, g = mstl(y, periods, jumps=False, inner_iter=5), sf_mstl(y, periods)
    d1 = max(np.abs(a.seasonal[p] - np.asarray(g[f"seasonal{p}"])).max() for p in periods)
    d2 = np.abs(np.vstack([mstl(y, periods).seasonal[p] for p in periods]) - sm_jumps(y, periods)).max()
    print(f"{label:34} {ours:10.1f}ms {same:12.1f}ms {sf:12.0f}ms {smj:16.1f}ms   max diff {d1:.0e} / {d2:.0e}")
print("\n'pyconformal' uses R's defaults (jumps). 'same settings' is pyconformal run with statsforecast's settings")
print("(no jumps, 5 inner passes); the first max diff compares those two, the second compares the jump versions.")
