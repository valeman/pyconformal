"""Multi-class learn-then-test: several one-vs-rest precision (or recall) targets at once.
The share of runs in which ANY class misses its target at the chosen threshold must be at most 1 - confidence_level."""
import sys, numpy as np, warnings
warnings.simplefilter("ignore")
from mapie.risk_control import MultiClassificationLTTController
from mapie.risk_control.risks import build_precision_ovr, build_recall_ovr
K = 3
def draw(n, rng):
    P = rng.dirichlet(np.ones(K) * 0.8, n); y = (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1); return P, np.minimum(y, K - 1)
big_P, big_y = draw(1_000_000, np.random.default_rng(123))
mx, am = big_P.max(1), big_P.argmax(1)
def truth(kind, lam):
    keep = mx >= lam; out = []
    for c in range(K):
        if kind == "precision": m = keep & (am == c); out.append((big_y[m] == c).mean() if m.any() else 1.0)
        else: m = big_y == c; out.append((keep[m] & (am[m] == c)).mean())
    return np.array(out)
R = int(sys.argv[1])
for kind, build, target in (("precision", build_precision_ovr, 0.6), ("precision", build_precision_ovr, 0.7), ("recall", build_recall_ovr, 0.3)):
    for n in (300, 1000):
        for fw in ("bonferroni_holm", "bonferroni"):
            bad = none = 0; lams = []; err = None
            for r in range(R):
                rng = np.random.default_rng(982451653 * r + n); P, y = draw(n, rng)
                try:
                    c = MultiClassificationLTTController(lambda X: np.asarray(X), risks=build(list(range(K))), target_level=[target] * K, confidence_level=0.9, classes=np.arange(K), fwer_method=fw)
                    c.calibrate(P, y); lam = c.best_predict_param
                except Exception as e:
                    err = f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"; break
                if lam is None or np.isnan(lam): none += 1; continue
                lams.append(float(lam)); bad += (truth(kind, float(lam)) < target).any()
            if err: print(f"{kind} >= {target}, n={n}, {fw}: ERROR {err}", flush=True); continue
            se = np.sqrt(0.09 / R)
            print(f"{kind:9s} >= {target} for all {K} classes, n={n:4d}, {fw:16s}: some class below target in {bad/R:.3f} of runs (allowed 0.10) {'VIOLATED' if bad/R > 0.1 + 3*se else 'ok'}; no threshold in {none} of {R}; median threshold {np.median(lams) if lams else float('nan'):.2f}", flush=True)
