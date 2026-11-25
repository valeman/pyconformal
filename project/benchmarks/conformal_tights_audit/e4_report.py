"""Tables from the e4 runs."""
import sys, glob, pickle, numpy as np
out = sys.argv[1]; names = ["diabetes", "ames", "abalone", "kin8nm", "grid_stability", "bike", "california", "superconduct"]
M = [("ct", "conformal-tights"), ("ct_abs", "  absolute branch only"), ("ct_rel", "  relative branch only"), ("pc_split", "pyconformal split"), ("pc_cqr", "pyconformal CQR")]
for cov in (0.9, 0.95):
    print(f"\nTarget coverage {cov}. Mean over 40 trials.")
    print(f"{'dataset':15s} {'n':>5s} {'L2 rows':>7s} | " + " | ".join(f"{lab.strip()[:22]:>22s}" for _, lab in M) + " | width ct / split / CQR | interval score ct / split / CQR | trials ct below target")
    ses = []
    for nm in names:
        try: res = pickle.load(open(f"{out}/e4_{nm}.pkl", "rb"))
        except FileNotFoundError: continue
        for n, tr in res.items():
            g = lambda k, j=0: np.array([t[k, cov][j] for t in tr]); ses.append(g("ct").std() / np.sqrt(len(tr)))
            print(f"{nm:15s} {n:5d} {tr[0]['l2']:7d} | " + " | ".join(f"{g(k).mean():22.3f}" for k, _ in M) +
                  f" | {g('ct',1).mean():9.4g} {g('pc_split',1).mean():9.4g} {g('pc_cqr',1).mean():9.4g} | {g('ct',2).mean():9.4g} {g('pc_split',2).mean():9.4g} {g('pc_cqr',2).mean():9.4g} | {np.mean(g('ct') < cov):.2f}   crossed {np.mean([t['ct_crossed', cov][0] for t in tr]):.4f}  time ct {g('ct',3).mean():.1f}s split {g('pc_split',3).mean():.1f}s")
    print("largest standard error of conformal-tights mean coverage:", round(max(ses), 4))
# summary by n
print("\nSummary over datasets (unweighted mean of dataset means), target 0.90 / 0.95")
for n in (300, 1000, 5000):
    rows = {k: [[], []] for k, _ in M}; cnt = 0
    for nm in names:
        try: res = pickle.load(open(f"{out}/e4_{nm}.pkl", "rb"))
        except FileNotFoundError: continue
        if n not in res: continue
        cnt += 1
        for k, _ in M:
            for j, cov in enumerate((0.9, 0.95)): rows[k][j].append(np.mean([t[k, cov][0] for t in res[n]]))
    print(f"n={n} ({cnt} datasets): " + "; ".join(f"{lab.strip()} {np.mean(rows[k][0]):.3f}/{np.mean(rows[k][1]):.3f}" for k, lab in M))
