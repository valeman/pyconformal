"""Weighted conformal predictive systems: pyconformal against crepes-weighted 0.1.3, the code released with
Jonkers, Van Wallendael, Duchateau and Van Hoecke (2024), and both against the definition and the guarantee.
Needs: pip install crepes-weighted crepes pandas."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from pyconformal import ConformalPredictiveSystem as P
from crepes_weighted import ConformalPredictiveSystem as W
from crepes import ConformalPredictiveSystem as C
g = np.random.default_rng(0)

print("1. Distribution function Q(y, tau), same random tau")
worst = 0.0
for trial in range(50):
    n, m = int(g.integers(5, 300)), 200
    res = np.round(g.normal(size=n), 1) if trial % 2 else g.normal(size=n); wc = g.uniform(0.1, 3, n); wt = g.uniform(0.1, 3, m)
    # with ties the predictions are zero, so that 'score equals threshold' means the same in both codes
    # (one compares y_pred + score with y, the other y - y_pred with the score, and rounding can differ)
    yh = np.zeros(m) if trial % 2 else g.normal(size=m); y = yh + (np.round(g.normal(size=m), 1) if trial % 2 else g.normal(size=m))
    np.random.seed(trial); theirs = W().fit(res, likelihood_ratios=wc).predict(yh, likelihood_ratios=wt, y=y)
    np.random.seed(trial); tau = np.random.rand(m)
    ours = P().calibrate(y_pred=np.zeros(n), y=res, weights=wc).predict_cdf(y_pred=yh, y=y, weights=wt, tau=tau)
    worst = max(worst, float(np.max(np.abs(np.ravel(theirs) - ours))))
print(f"   largest difference over 50 problems (half of them with ties): {worst:.2e}")

print("2. Quantiles with equal weights: does each reduce to the unweighted system (crepes 0.9.1)?")
n, m = 99, 5; res = g.normal(size=n); yh = np.zeros(m); srt = np.sort(res)
rank = lambda v: int(np.searchsorted(srt, v[0]) + 1) if np.isfinite(v[0]) else None
for lo, hi in ((5, 95), (2.5, 97.5), (10, 90)):
    c = C().fit(res).predict(yh, lower_percentiles=lo, higher_percentiles=hi)
    w = W().fit(res, likelihood_ratios=np.ones(n)).predict(yh, likelihood_ratios=np.ones(m), lower_percentiles=lo, higher_percentiles=hi)
    p = P().calibrate(y_pred=np.zeros(n), y=res, weights=np.ones(n))
    pl, ph = p.predict_quantile(y_pred=yh, q=lo / 100, weights=1.0, side="lower"), p.predict_quantile(y_pred=yh, q=hi / 100, weights=1.0, side="higher")
    print(f"   n = 99, percentiles {lo}/{hi}: rank of lower and upper end among the sorted scores: "
          f"crepes {rank(c[:, 0])}/{rank(c[:, 1])}, pyconformal {rank(pl)}/{rank(ph)}, crepes-weighted {rank(w[:, 0])}/{rank(w[:, 1])}")

print("3. Guarantee under a known covariate shift (calibration x ~ N(0,1), test x ~ N(1,1), weights = true likelihood ratio)")
lr = lambda x: np.exp(x - 0.5)
for n in (20, 50, 200, 1000):
    R = 4000 if n <= 200 else 1000; m = 200; out = {k: [] for k in ("ours", "theirs")}; wid = {k: [] for k in out}; inf = {k: 0 for k in out}; below = {k: [] for k in out}; above = {k: [] for k in out}
    for _ in range(R):
        xc, xt = g.normal(size=n), g.normal(1, 1, m); yc, yt = (1 + np.abs(xc)) * g.normal(size=n), (1 + np.abs(xt)) * g.normal(size=m)
        io = P().calibrate(y_pred=np.zeros(n), y=yc, weights=lr(xc)).predict_interval(y_pred=np.zeros(m), alpha=0.1, weights=lr(xt))
        it = W().fit(yc, likelihood_ratios=lr(xc)).predict(np.zeros(m), likelihood_ratios=lr(xt), lower_percentiles=5, higher_percentiles=95)
        for k, iv in (("ours", io), ("theirs", it)):
            out[k].append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean()); below[k].append((yt < iv[:, 0]).mean()); above[k].append((yt > iv[:, 1]).mean())
            f = np.isfinite(iv).all(axis=1); inf[k] += int((~f).sum()); wid[k].append(np.median((iv[:, 1] - iv[:, 0])[f]) if f.any() else np.nan)
    for k in out:
        print(f"   n = {n:4d} {k:7s}: coverage {np.mean(out[k]):.4f} (s.e. {np.std(out[k]) / np.sqrt(R):.4f}), below {np.mean(below[k]):.4f}, above {np.mean(above[k]):.4f}, "
              f"median width of finite intervals {np.nanmean(wid[k]):.3f}, infinite intervals {inf[k] / (R * m):.1%}")
