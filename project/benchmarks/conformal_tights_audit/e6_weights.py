"""conformal-tights 0.5.0 and sample weights.
One fifth of the rows are corrupted (noise ten times larger) and given weight 0; the rest have weight 1.
If weights were honoured everywhere, fitting with these weights would behave like fitting on the clean rows only.
Test data are clean. Defaults, n = 6,000 rows (4,800 clean), 90% and 95% intervals, R trials.
Also reported: the level-2 shift applied, against the shift a weighted quantile would give."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from conformal_tights import ConformalCoherentQuantileRegressor as CT
R = int(sys.argv[1]); out = {k: [] for k in ("weighted", "clean rows only", "unweighted, all rows")}; shifts = []
def wq(v, w, q):
    o = np.argsort(v); c = np.cumsum(w[o]) / w.sum(); return v[o][np.searchsorted(c, q)]
for s in range(R):
    rng = np.random.default_rng(s); n = 6000
    X = rng.normal(size=(n, 5)); bad = rng.random(n) < 0.2
    y = 10 + 2 * X[:, 0] + np.sin(2 * X[:, 1]) + (0.5 + np.abs(X[:, 2])) * rng.normal(size=n) * np.where(bad, 10, 1)
    Xt = rng.normal(size=(4000, 5)); yt = 10 + 2 * Xt[:, 0] + np.sin(2 * Xt[:, 1]) + (0.5 + np.abs(Xt[:, 2])) * rng.normal(size=4000)
    w = np.where(bad, 0.0, 1.0)
    fits = {"weighted": CT(random_state=s).fit(X, y, sample_weight=w), "clean rows only": CT(random_state=s).fit(X[~bad], y[~bad]), "unweighted, all rows": CT(random_state=s).fit(X, y)}
    for k, m in fits.items():
        row = []
        for cov in (0.9, 0.95):
            iv = np.asarray(m.predict_interval(Xt, coverage=cov)); row += [np.mean((iv[:, 0] <= yt) & (yt <= iv[:, 1])), np.mean(iv[:, 1] - iv[:, 0])]
        out[k].append(row)
    m = fits["weighted"]; q = (0.050000000000000044, 0.95); q = ((1 - 0.9) / 2, 1 - (1 - 0.9) / 2)
    l1 = m.conformal_l1_["Δŷ"][q]; r = -m.residuals_calib_l2_[:, None] - l1.predict(m.ŷ_calib_l2_nonconformity_)
    w2 = m.sample_weight_calib_l2_
    shifts.append([np.quantile(r[:, 0], q[0]), wq(r[:, 0], w2, q[0]), np.quantile(r[:, 1], q[1]), wq(r[:, 1], w2, q[1]), np.mean(w2 == 0)])
    print(f"trial {s} done", flush=True)
print(f"{'fit':24s} cov@90  width@90  cov@95  width@95   (mean of {R} trials; se of coverage below)")
for k, v in out.items():
    v = np.array(v); print(f"{k:24s} {v[:,0].mean():.4f}  {v[:,1].mean():7.3f}   {v[:,2].mean():.4f}  {v[:,3].mean():7.3f}    se {v[:,0].std()/np.sqrt(R):.4f} {v[:,2].std()/np.sqrt(R):.4f}")
s = np.array(shifts).mean(0)
print(f"level-2 shift, 90% interval, absolute residuals, weighted fit: lower end unweighted {s[0]:.3f} (weighted would be {s[1]:.3f}); upper end unweighted {s[2]:.3f} (weighted would be {s[3]:.3f}); share of zero-weight rows in level 2: {s[4]:.2f}")
