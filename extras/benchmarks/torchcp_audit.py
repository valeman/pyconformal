"""TorchCP audit: threshold exactness and coverage of split conformal classification on i.i.d. data."""
import math, sys, warnings; warnings.filterwarnings("ignore")
from fractions import Fraction
import numpy as np, torch
from importlib.metadata import version
from torchcp.utils.common import calculate_conformal_value
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import LAC, APS, RAPS, SAPS, TOPK
print("torchcp", version("torchcp"))
# 1. threshold exactness
rng = np.random.default_rng(0)
alphas = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.9]
ok = wide = narrow = 0; ex = ""
for n in list(range(1, 121)) + [199, 200, 999, 1000]:
    for a in alphas:
        s = rng.exponential(size=n)
        k = math.ceil((n + 1) * (1 - Fraction(repr(a)))); e = math.inf if k > n else float(np.sort(s)[k - 1])
        h = float(calculate_conformal_value(torch.tensor(s), a))
        if (math.isinf(e) and math.isinf(h)) or (not math.isinf(e) and abs(h - e) < 1e-9): ok += 1
        elif h > e: wide += 1; ex = ex or f"n={n}, alpha={a}: got {h:.4g}, exact {e:.4g}"
        else: narrow += 1; ex = ex or f"n={n}, alpha={a}: got {h:.4g}, exact {e:.4g}"
print(f"threshold: exact {ok}, too wide {wide}, too narrow {narrow} of 1240. {ex}")
# 2. coverage
def data(rng, n, K=10):
    y = rng.integers(0, K, size=n); logits = rng.normal(size=(n, K)) * 1.5 + 2.5 * np.eye(K)[y]
    return torch.tensor(logits, dtype=torch.float32), torch.tensor(y)
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
print(f"\ncoverage, {R} repetitions, 2,000 test points each, 10 classes")
for level in (0.9, 0.8):
    for name, mk in [("LAC", lambda: LAC()), ("APS randomized", lambda: APS()), ("APS not randomized", lambda: APS(randomized=False)),
                     ("RAPS randomized", lambda: RAPS(penalty=0.1, kreg=1)), ("RAPS not randomized", lambda: RAPS(randomized=False, penalty=0.1, kreg=1)),
                     ("SAPS randomized", lambda: SAPS()), ("SAPS not randomized", lambda: SAPS(randomized=False)), ("TOPK", lambda: TOPK())]:
        for n_cal in (20, 100, 1000):
            covs, sizes = [], []
            try:
                for r in range(R):
                    rng = np.random.default_rng(1000 * r + n_cal); torch.manual_seed(r)
                    lc, yc = data(rng, n_cal); lt, yt = data(rng, 2000)
                    p = SplitPredictor(mk(), model=None, alpha=1 - level)
                    p.calculate_threshold(lc, yc)
                    sets = p.predict_with_logits(lt)
                    sets = sets if torch.is_tensor(sets) else torch.tensor(np.asarray(sets))
                    covs.append(sets[torch.arange(2000), yt].float().mean().item()); sizes.append(sets.float().sum(1).mean().item())
                m, se = np.mean(covs), np.std(covs, ddof=1) / np.sqrt(R)
                print(f"{'FAIL' if m < level - 3 * se else 'ok':4} {name:20} n_cal={n_cal:5} target>={level:.2f} coverage {m:.4f} (se {se:.4f}) mean set size {np.mean(sizes):.2f}", flush=True)
            except Exception as e:
                print(f"ERR  {name:20} n_cal={n_cal}: {type(e).__name__}: {str(e)[:90]}")
