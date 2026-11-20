import time, warnings, math
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(2)
T0, T = 500, 2000
y = np.cumsum(rng.normal(size=T0 + T + 10)) * 0.1 + rng.standard_t(3, size=T0 + T + 10)
L = 5
X = np.column_stack([y[i:len(y) - L + i] for i in range(L)]); Y = y[L:]
Xtr, Ytr, Xte, Yte = X[:T0], Y[:T0], X[T0:T0 + T], Y[T0:T0 + T]
model = LinearRegression().fit(Xtr[: T0 // 2], Ytr[: T0 // 2])

# MAPIE ACI (TimeSeriesRegressor, method="aci"), one step at a time
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
m = TimeSeriesRegressor(model, method="aci", cv="prefit")
m.fit(Xtr[T0 // 2:], Ytr[T0 // 2:])
t0 = time.perf_counter(); cov = []
for t in range(T):
    _, iv = m.predict(Xte[t:t + 1], confidence_level=0.9, allow_infinite_bounds=True)
    cov.append(iv[0, 0, 0] <= Yte[t] <= iv[0, 1, 0])
    m.adapt_conformal_inference(Xte[t:t + 1], Yte[t:t + 1], gamma=0.01, confidence_level=0.9)
tm = time.perf_counter() - t0
print(f"MAPIE ACI: {T} steps in {tm:.2f} s ({1e3 * tm / T:.2f} ms/step), coverage {np.mean(cov):.3f}")

# Reference ACI (Gibbs & Candes 2021) on the same scores, plain NumPy loop over a sorted buffer
import bisect
pred = model.predict(Xte); cal = list(np.sort(np.abs(Ytr[T0 // 2:] - model.predict(Xtr[T0 // 2:]))))
t0 = time.perf_counter(); a_t = 0.1; cov = []
for t in range(T):
    n = len(cal); k = math.ceil((n + 1) * (1 - a_t)) if a_t > 0 else n + 1
    q = math.inf if k > n else (cal[max(k, 1) - 1])
    err = abs(Yte[t] - pred[t]) > q; cov.append(not err)
    a_t += 0.01 * (0.1 - err)
    bisect.insort(cal, abs(Yte[t] - pred[t]))
tr = time.perf_counter() - t0
print(f"Reference ACI (pure Python, sorted buffer): {T} steps in {tr:.3f} s ({1e6 * tr / T:.1f} us/step), coverage {np.mean(cov):.3f}")
print(f"speed ratio: {tm / tr:.0f}x")
