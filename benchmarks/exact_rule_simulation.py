"""Coverage of the plain interpolated quantile (np.quantile, as statsforecast/mlforecast use)
against the exact finite-sample rule, for i.i.d. errors at nominal 90%."""
import math
import numpy as np
rng = np.random.default_rng(0); R = 200_000; level = 0.9
print("n_windows  plain np.quantile  exact rule")
for n in [2, 3, 5, 9, 10, 20, 50]:
    cs = np.abs(rng.standard_normal((R, n))); new = np.abs(rng.standard_normal(R))
    q = np.quantile(cs, level, axis=1); k = math.ceil((n + 1) * level)
    qe = np.sort(cs, axis=1)[:, k - 1] if k <= n else np.full(R, np.inf)
    print(f"{n:9d}  {np.mean(new <= q):17.3f}  {np.mean(new <= qe):10.3f}{'  (interval infinite)' if k > n else ''}")
