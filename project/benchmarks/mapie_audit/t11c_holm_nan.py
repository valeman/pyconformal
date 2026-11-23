import numpy as np, warnings
from mapie.risk_control import BinaryClassificationController
from mapie.risk_control.fwer_control import control_fwer
warnings.simplefilter("ignore")
f = lambda X: np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]])
# (1) the smallest example: one NaN p-value among clearly significant ones
p = np.array([1e-9, 1e-9, 1e-9, np.nan])
print("p-values", p, " delta 0.1")
for m in ("bonferroni", "bonferroni_holm"): print(f"  {m:16s} rejects indices {control_fwer(p, 0.1, fwer_method=m).tolist()}")
# (2) in the controller: runs with no threshold, split by whether any p-value is NaN
R = 300; tab = {}
for r in range(R):
    rng = np.random.default_rng(15485863 * r + 2000); pr = rng.beta(2, 2, 2000); y = (rng.random(2000) < pr).astype(int)
    out = []
    for fw in ("bonferroni_holm", "bonferroni"):
        c = BinaryClassificationController(f, "precision", target_level=0.6, confidence_level=0.9, fwer_method=fw); c.calibrate(pr.reshape(-1, 1), y)
        out.append(c.best_predict_param is None or bool(np.isnan(c.best_predict_param)))
    key = (bool(np.isnan(c.p_values).any()), out[0], out[1]); tab[key] = tab.get(key, 0) + 1
print("\nprecision >= 0.6, n = 2000, 300 runs  (any NaN p-value, Holm returns nothing, Bonferroni returns nothing): count")
for k, v in sorted(tab.items()): print("  ", k, v)
