import numpy as np
from importlib.metadata import version
from crepes import ConformalRegressor
print("crepes", version("crepes"))
for n in (9, 19, 29, 39, 49, 99):
    res = np.arange(1.0, n + 1)                       # residuals 1..n, so the k-th smallest is k
    cr = ConformalRegressor().fit(res)
    lo, hi = cr.predict_int(np.array([0.0]), confidence=0.9)[0]
    exact = int(np.ceil(round((n + 1) * 0.9, 9)))     # ceil((n + 1)(1 - alpha))
    print(f"n={n}: (n+1)*0.9 = {(n + 1) * 0.9!r}, half-width used {hi:.0f}, rank ceil((n+1)*0.9) = {exact}")
