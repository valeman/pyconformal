"""Exchangeability tests: on i.i.d. data each test may reject in at most test_level of runs (5%). Then one drifting case, to see that they can reject at all."""
import sys, time, numpy as np, warnings
from common import *
from mapie.regression import SplitConformalRegressor
from mapie.exchangeability_testing import OnlineMartingaleTest, PValuePermutationTest, SequentialMonteCarloTest
warnings.simplefilter("ignore")
R, T = int(sys.argv[1]), int(sys.argv[2])
def model(rng):
    X = np.zeros((200, 1)); y = rng.normal(size=200)
    return SplitConformalRegressor(Const(0.0), confidence_level=0.9, prefit=True).conformalize(X, y)
def data(kind, rng, T):
    e = rng.normal(size=T)
    if kind == "drift": e = e * np.linspace(1, 4, T)
    if kind == "ties": e = np.round(e)            # heavy ties in the scores
    return np.zeros((T, 1)), e
tests = {
    "jumper martingale": lambda m, s: OnlineMartingaleTest(mapie_estimator=m, test_method="jumper_martingale", warn=False, random_state=s),
    "plug-in martingale": lambda m, s: OnlineMartingaleTest(mapie_estimator=m, test_method="plugin_martingale", warn=False, random_state=s),
    "p-value permutation": lambda m, s: PValuePermutationTest(mapie_estimator=m, random_state=s, num_permutations=200, warn=False),
    "sequential MC, aggressive": lambda m, s: SequentialMonteCarloTest("aggressive", mapie_estimator=m, random_state=s, num_permutations=200, warn=False),
    "sequential MC, binomial": lambda m, s: SequentialMonteCarloTest("binomial", mapie_estimator=m, random_state=s, num_permutations=200, warn=False),
    "sequential MC, binomial mixture": lambda m, s: SequentialMonteCarloTest("binomial_mixture", mapie_estimator=m, random_state=s, num_permutations=200, warn=False),
}
for kind in ("iid", "ties", "drift"):
    for name, make in tests.items():
        rej = none = 0; t0 = time.time(); err = None
        rr = R if kind != "drift" else max(40, R // 10)
        for r in range(rr):
            rng = np.random.default_rng(49979687 * r + 11); m = model(rng); X, y = data(kind, rng, T)
            try:
                t = make(m, r)
                t = t.update(X, y) if hasattr(t, "update") else t.run(X, y)
                d = t.is_exchangeable
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:110]}"; break
            none += d is None; rej += d is False
        if err: print(f"{kind:5s} {name:32s} ERROR {err}", flush=True); continue
        se = np.sqrt(0.05 * 0.95 / rr)
        v = ("ABOVE 5%" if rej / rr > 0.05 + 3 * se else "ok") if kind != "drift" else ""
        print(f"{kind:5s} {name:32s} rejects in {rej/rr:.3f} of {rr} runs; undecided {none}  {v}   ({time.time()-t0:.0f}s)", flush=True)
