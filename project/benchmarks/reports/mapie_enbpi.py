import cProfile, pstats, time, io
import numpy as np
from importlib.metadata import version
from sklearn.linear_model import LinearRegression
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
print("mapie", version("mapie"))
rng = np.random.default_rng(0)
n_train, n_test = 1000, 500
X = rng.normal(size=(n_train + n_test, 5)); y = X @ np.arange(1.0, 6.0) + rng.normal(size=n_train + n_test)
m = TimeSeriesRegressor(LinearRegression(), method="enbpi", agg_function="mean",
                        cv=BlockBootstrap(n_resamplings=30, n_blocks=10, overlapping=False, random_state=0))
m.fit(X[:n_train], y[:n_train])

def stream():
    hits = 0
    for i in range(n_train, n_train + n_test):
        _, pis = m.predict(X[i:i + 1], confidence_level=0.9, ensemble=True)
        hits += pis[0, 0, 0] <= y[i] <= pis[0, 1, 0]
        m.update(X[i:i + 1], y[i:i + 1], ensemble=True)
    return hits / n_test

t = time.perf_counter(); cov = stream(); el = time.perf_counter() - t; plain = el
m.fit(X[:n_train], y[:n_train]); pr = cProfile.Profile(); pr.enable(); stream(); pr.disable()
print(f"{el / n_test * 1e3:.2f} ms per step (predict + update), coverage {cov:.3f}")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14); print("\n".join(l for l in s.getvalue().splitlines()[4:] if l.strip())[:2600])
