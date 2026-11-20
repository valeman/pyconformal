"""Quantile forest: pyconformal against quantile-forest (Zillow).

Same settings on both sides: 100 trees, min_samples_leaf=5, a third of the
features per split, all cores. Zillow is run in its default mode (one stored
sample per leaf, an approximation) and in its exact mode (all samples).
Accuracy: mean pinball loss over the levels, lower is better.
"""
import sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from pyconformal import QuantileForest
from quantile_forest import RandomForestQuantileRegressor as Zillow

rng = np.random.default_rng(0)
def data(n, d=10):
    X = rng.uniform(-1, 1, (n, d))
    return X, np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + (0.3 + 0.5 * np.abs(X[:, 2])) * rng.standard_exponential(n)
def pinball(y, q, t):
    e = y[:, None] - q; return np.mean(np.maximum(t * e, (t - 1) * e))
def timed(f):
    t = time.perf_counter(); r = f(); return time.perf_counter() - t, r

QuantileForest(2).fit(*data(300)).predict(data(10)[0], quantiles=[0.5])  # compile
QuantileForest(2, max_samples_leaf=1).fit(*data(300)).predict(data(10)[0], quantiles=[0.5])
print(f"quantile-forest {version('quantile-forest')}; 20,000 test rows")
sizes = [int(a) for a in sys.argv[1:]] or [10_000, 100_000]
for levels, label in ((np.array([0.05, 0.5, 0.95]), "3 quantile levels"), (np.linspace(0.01, 0.99, 99), "99 quantile levels")):
    print(f"\n{label}")
    print(f"{'training rows':>14} {'library':30} {'fit s':>7} {'predict s':>10} {'pinball':>8}")
    for n in sizes:
        X, y = data(n); Xt, yt = data(20_000)
        for one, label2 in ((None, "pyconformal"), (1, "pyconformal, 1 sample per leaf")):
          tf, m = timed(lambda: QuantileForest(100, min_samples_leaf=5, max_features=1 / 3, random_state=0, max_samples_leaf=one).fit(X, y))
          tp, q = timed(lambda: m.predict(Xt, quantiles=levels))
          print(f"{n if one is None else '':>14} {label2:30} {tf:7.2f} {tp:10.2f} {pinball(yt, q, levels):8.4f}")
        for msl, name in ((1, "zillow default (approximate)"), (None, "zillow exact")):
            tf, z = timed(lambda: Zillow(100, min_samples_leaf=5, max_features=1 / 3, random_state=0, n_jobs=-1, max_samples_leaf=msl).fit(X, y))
            tp, q = timed(lambda: z.predict(Xt, quantiles=list(levels)))
            print(f"{'':>14} {name:30} {tf:7.2f} {tp:10.2f} {pinball(yt, q, levels):8.4f}")
