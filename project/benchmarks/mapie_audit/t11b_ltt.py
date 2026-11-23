import numpy as np, warnings, sys
from mapie.risk_control import BinaryClassificationController
warnings.simplefilter("ignore")
big = np.random.default_rng(99).beta(2, 2, 2_000_000)
def true_precision(t): m = big >= t; return big[m].mean()   # MAPIE predicts positive when p >= threshold
def true_recall(t): return big[big >= t].sum() / big.sum()
R = int(sys.argv[1])
f = lambda X: np.column_stack([1 - np.asarray(X)[:, 0], np.asarray(X)[:, 0]])
for risk, target, truth in (("precision", 0.6, true_precision), ("precision", 0.7, true_precision), ("recall", 0.8, true_recall), ("recall", 0.9, true_recall)):
    for n in (200, 500, 2000):
        for fw in ("bonferroni_holm", "bonferroni", "fixed_sequence"):
            bad = none = 0; thr = []
            for r in range(R):
                rng = np.random.default_rng(15485863 * r + n); p = rng.beta(2, 2, n); y = (rng.random(n) < p).astype(int)
                try:
                    c = BinaryClassificationController(f, risk, target_level=target, confidence_level=0.9, fwer_method=fw); c.calibrate(p.reshape(-1, 1), y)
                except Exception as e:
                    print("ERROR", risk, fw, type(e).__name__, str(e)[:120]); none = -1; break
                t = c.best_predict_param
                if t is None or np.isnan(t): none += 1; continue
                thr.append(float(t)); bad += truth(float(t)) < target
            se = np.sqrt(0.09 / R)
            print(f"{risk:9s} >= {target} n={n:4d} {fw:16s}: true {risk} below target in {bad/R:.3f} of runs (allowed 0.10) {'VIOLATED' if bad/R > 0.1 + 3*se else 'ok'}; no threshold in {none}; median threshold {np.median(thr) if thr else float('nan'):.2f}", flush=True)
