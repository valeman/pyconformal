"""CV+ (10 folds): pyconformal against MAPIE, same folds-free setting.
Linear model, so model training is negligible and the timing shows the
conformal step. Seconds, best of 3.
"""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pyconformal as pc
from mapie.regression import CrossConformalRegressor as Mapie

def best(f, k=3):
    b, r = 1e9, None
    for _ in range(k):
        s = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - s)
    return b, r
rng = np.random.default_rng(0)
def data(n, d=10):
    X = rng.normal(size=(n, d)); return X, X @ np.arange(1, d + 1) + rng.standard_t(4, size=n)
pc.CrossConformalRegressor(LinearRegression(), cv=3).fit(*data(100)).predict_interval(data(5)[0])  # compile
print(f"MAPIE {version('mapie')}; CV+ with 10 folds, 90% target")
print(f"{'train':>8} {'test':>8} {'library':12} {'fit s':>7} {'predict s':>10} {'coverage':>9} {'width':>7}")
for n, nt in ((2_000, 20_000), (5_000, 20_000), (20_000, 20_000), (100_000, 20_000)):
    X, y = data(n); Xt, yt = data(nt)
    tf, cp = best(lambda: pc.CrossConformalRegressor(LinearRegression(), alpha=0.1, cv=10, random_state=0).fit(X, y))
    tp, iv = best(lambda: cp.predict_interval(Xt))
    print(f"{n:>8,} {nt:>8,} {'pyconformal':12} {tf:7.2f} {tp:10.3f} {((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean():9.4f} {np.mean(iv[:, 1] - iv[:, 0]):7.3f}")
    try:
        if n * nt > 1.5e8: raise MemoryError("not run: builds a train x test matrix (killed for memory at 20,000 x 20,000 here)")
        tf, m = best(lambda: Mapie(LinearRegression(), confidence_level=0.9, method="plus", cv=KFold(10, shuffle=True, random_state=0)).fit_conformalize(X, y), 1)
        tp, r = best(lambda: m.predict_interval(Xt), 1); mv = r[1][:, :, 0]
        print(f"{'':>8} {'':>8} {'MAPIE':12} {tf:7.2f} {tp:10.3f} {((mv[:, 0] <= yt) & (yt <= mv[:, 1])).mean():9.4f} {np.mean(mv[:, 1] - mv[:, 0]):7.3f}")
    except Exception as e:
        print(f"{'':>8} {'':>8} {'MAPIE':12} {type(e).__name__}: {str(e)[:90]}")
