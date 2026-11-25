"""Finding 4 on more fits: the level-2 shift the calibration data ask for against the shift applied after clipping.
usage: e8_clip.py <mode> <fits> [data dir]   mode: synth5000 | synth2000 | real"""
import sys, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from conformal_tights import ConformalCoherentQuantileRegressor as CT
mode, R = sys.argv[1], int(sys.argv[2]); Q = (0.05, 0.25, 0.5, 0.75, 0.95)
def synth(n, rng):
    X = rng.normal(size=(n, 5)); return X, 10 + 2 * X[:, 0] + np.sin(2 * X[:, 1]) + (0.5 + np.abs(X[:, 2])) * rng.standard_t(4, size=n)
def one(X, y, Xt, yt, seed):
    m = CT(random_state=seed).fit(X, y); t0 = time.time(); P = np.asarray(m.predict_quantiles(Xt, quantiles=Q)); dt = time.time() - t0
    res = [dt, (yt[:, None] <= P).mean(0), np.mean(np.any(np.diff(P, axis=1) < 0, axis=1))]
    for tt in ("Δŷ", "Δŷ/ŷ"):
        l1 = m.conformal_l1_[tt][Q]; den = np.maximum(np.abs(m.ŷ_calib_l2_), np.finfo(float).eps) if "/" in tt else 1
        r = (-m.residuals_calib_l2_ / den)[:, None] - l1.predict(m.ŷ_calib_l2_nonconformity_)
        want = np.array([np.quantile(r[:, j], q) for j, q in enumerate(Q)]); got = np.asarray(m.conformal_l2_[tt][Q])
        sc = np.std(-m.residuals_calib_l2_ / den)
        res.append((want / sc, got / sc))
    return res
runs = []
if mode.startswith("synth"):
    n = int(mode[5:])
    for s in range(R):
        rng = np.random.default_rng(50_000 + s); X, y = synth(n, rng); Xt, yt = synth(3000, rng); runs.append(one(X, y, Xt, yt, s)); print(f"{mode} fit {s}: {runs[-1][0]:.0f} s", flush=True)
else:
    for name in ("california", "bike", "kin8nm", "superconduct"):
        d = np.load(f"{sys.argv[3]}/{name}.npz"); X, y = d["X"], d["y"]
        for s in range(R):
            p = np.random.default_rng(s).permutation(len(y)); runs.append(one(X[p[:5000]], y[p[:5000]], X[p[5000:8000]], y[p[5000:8000]], s)); print(f"{name} fit {s}: {runs[-1][0]:.0f} s", flush=True)
print(f"\n{mode}: {len(runs)} fits, 5 levels. Shifts are in units of the standard deviation of the level-2 residuals.")
print("level                               ", *[f"{q:7.2f}" for q in Q])
for i, tt in ((3, "absolute residuals"), (4, "relative residuals")):
    W = np.array([r[i][0] for r in runs]); G = np.array([r[i][1] for r in runs])
    print(f"{tt}: mean |shift asked|   ", *[f"{v:7.4f}" for v in np.abs(W).mean(0)])
    print(f"{tt}: mean |shift applied| ", *[f"{v:7.4f}" for v in np.abs(G).mean(0)])
    print(f"{tt}: share of fits clipped", *[f"{v:7.2f}" for v in (np.abs(W - G) > 1e-9 * (1 + np.abs(W))).mean(0)])
    print(f"{tt}: share of fits with applied shift exactly 0", *[f"{v:7.2f}" for v in (G == 0).mean(0)])
F = np.array([r[1] for r in runs])
print("observed frequency of y <= quantile ", *[f"{v:7.4f}" for v in F.mean(0)])
print("largest |frequency - level| in a fit", *[f"{v:7.4f}" for v in np.abs(F - np.array(Q)).max(0)])
print(f"share of test points with crossing quantiles: mean {np.mean([r[2] for r in runs]):.4f}, max {np.max([r[2] for r in runs]):.4f}")
print(f"first predict_quantiles call: median {np.median([r[0] for r in runs]):.0f} s, max {np.max([r[0] for r in runs]):.0f} s")
