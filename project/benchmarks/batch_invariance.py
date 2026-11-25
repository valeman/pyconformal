"""pyconformal APS, RAPS and SAPS: the same calibrated object asked for sets for all test rows in one call and one
row per call. Same probability model as puncc_audit/p0_raps_rows.py (10 classes, Dirichlet(0.3) probabilities,
labels drawn from them), 1,000 calibration and 1,000 test rows per split, 300 splits, 90% target.
The randomisation is seeded, so the two ways of calling must return the same sets."""
import numpy as np
from pyconformal import SplitConformalClassifier
K, R = 10, 300
def draw(g, n):
    P = g.dirichlet(np.ones(K) * 0.3, n); c = P.cumsum(axis=1); y = (g.random(n)[:, None] > c).sum(axis=1).clip(max=K - 1); return P, y
for score in ("aps", "raps", "saps"):
    cov = []; changed = 0; total = 0; size = []
    for r in range(1000, 1000 + R):
        g = np.random.default_rng(r); Pc, yc = draw(g, 1000); Pt, yt = draw(g, 1000)
        a = SplitConformalClassifier(score=score, alpha=0.1, random_state=r).calibrate(proba=Pc, y=yc).predict_set(proba=Pt)
        m = SplitConformalClassifier(score=score, alpha=0.1, random_state=r).calibrate(proba=Pc, y=yc)
        b = np.vstack([m.predict_set(proba=Pt[i:i + 1]) for i in range(1000)])
        changed += int((a != b).sum()); total += a.size
        cov.append(a[np.arange(1000), yt].mean()); size.append(a.sum(axis=1).mean())
    cov = np.array(cov)
    print(f"{score:5s} coverage {cov.mean():.4f} (s.e. {cov.std(ddof=1) / np.sqrt(R):.4f}), mean set size {np.mean(size):.2f}; "
          f"membership decisions that differ between one call and one row per call: {changed} of {total}")
