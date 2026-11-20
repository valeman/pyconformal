"""ACI: pyconformal against MAPIE on the same task. A fixed (prefit) linear forecaster,
then T online steps: predict a 90% interval, observe the outcome, update."""
import time, warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
from importlib.metadata import version
from sklearn.linear_model import LinearRegression
from mapie.regression import TimeSeriesRegressor
from pyconformal import AdaptiveConformalInference
from pyconformal._online_kernel import HAVE_NUMBA
print("mapie", version("mapie"), "| numba available:", HAVE_NUMBA)
rng = np.random.default_rng(2); L, T0 = 5, 500
def make(T):
    n = T0 + T + L
    y = np.cumsum(rng.normal(size=n)) * 0.1 + rng.standard_t(3, size=n) * np.where(np.arange(n) > n // 2, 3.0, 1.0)
    X = np.column_stack([y[i:n - L + i] for i in range(L)]); Y = y[L:]
    return X[:T0], Y[:T0], X[T0:], Y[T0:]
AdaptiveConformalInference().run(np.ones(3))  # compile once
print(f"\n{'steps':>9} {'MAPIE s':>9} {'ms/step':>8} {'ours s':>9} {'us/step':>8} {'ours numpy s':>12} {'speed-up':>9}  coverage MAPIE / ours")
for T in (2_000, 20_000, 1_000_000):
    Xtr, Ytr, Xte, Yte = make(T)
    h = T0 // 2
    model = LinearRegression().fit(Xtr[:h], Ytr[:h])
    # ours: one vectorised model call, then the replay
    def ours(nb):
        t0 = time.perf_counter()
        aci = AdaptiveConformalInference(alpha=0.1, gamma=0.01).calibrate(y_pred=model.predict(Xtr[h:]), y=Ytr[h:])
        res = aci.run(y_pred=model.predict(Xte), y=Yte, use_numba=nb)
        return time.perf_counter() - t0, res.coverage
    to, co = min(ours(True) for _ in range(3)) if HAVE_NUMBA else (float("nan"), float("nan"))
    tp, cp = ours(False)
    if T <= 20_000:
        m = TimeSeriesRegressor(model, method="aci", cv="prefit"); m.fit(Xtr[h:], Ytr[h:])
        t0 = time.perf_counter(); hit = 0
        for t in range(T):
            _, iv = m.predict(Xte[t:t + 1], confidence_level=0.9, allow_infinite_bounds=True)
            hit += iv[0, 0, 0] <= Yte[t] <= iv[0, 1, 0]
            m.adapt_conformal_inference(Xte[t:t + 1], Yte[t:t + 1], gamma=0.01, confidence_level=0.9)
        tm = time.perf_counter() - t0
        print(f"{T:9,} {tm:9.2f} {1e3 * tm / T:8.2f} {to:9.4f} {1e6 * to / T:8.2f} {tp:12.3f} {tm / to:8.0f}x  {hit / T:.3f} / {co:.3f}")
    else:
        print(f"{T:9,} {'not run':>9} {'':>8} {to:9.4f} {1e6 * to / T:8.2f} {tp:12.3f} {'':>9}  - / {co:.3f}")
