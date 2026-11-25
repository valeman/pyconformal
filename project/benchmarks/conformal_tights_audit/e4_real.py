"""conformal-tights 0.5.0 with its defaults ("auto": XGBoost defaults) on public datasets, side by side with
pyconformal on the same rows. Each trial: n rows drawn at random for fitting, up to 5,000 other rows for testing.
pyconformal gets the same n rows, split 70/30 with the same calibration size rule as conformal-tights.
usage: e4_real.py <data dir> <dataset> <trials> <out dir>"""
import sys, time, warnings, pickle, numpy as np
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from conformal_tights import ConformalCoherentQuantileRegressor as CT
from pyconformal import SplitConformalRegressor, ConformalizedQuantileRegressor
ddir, name, R, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = np.load(f"{ddir}/{name}.npz"); X, y = d["X"], d["y"]
COV = (0.9, 0.95)
def score(lo, hi, yt, cov):
    a = 1 - cov; w = hi - lo
    wink = w + 2 / a * np.maximum(lo - yt, 0) + 2 / a * np.maximum(yt - hi, 0)
    return [np.mean((lo <= yt) & (yt <= hi)), np.mean(w), np.mean(wink)]
def trial(n, seed):
    rng = np.random.default_rng(seed); p = rng.permutation(len(y))
    tr, te = p[:n], p[n:n + 5000]; Xa, ya, Xt, yt = X[tr], y[tr], X[te], y[te]
    res = {}
    t0 = time.time(); m = CT(random_state=seed, estimator=XGBRegressor(n_jobs=1), nonconformity_estimator=XGBRegressor(n_jobs=1)).fit(Xa, ya); tf = time.time() - t0
    yh = m.base_estimator_.predict(Xt); Z = m.nonconformity_estimator_.predict(Xt)
    for cov in COV:
        t0 = time.time(); iv = np.asarray(m.predict_interval(Xt, coverage=cov)); tp = time.time() - t0
        q = ((1 - cov) / 2, 1 - (1 - cov) / 2)
        a = yh[:, None] + m.conformal_l1_["Δŷ"][q].predict(Z) + m.conformal_l2_["Δŷ"][q]
        r = yh[:, None] + np.abs(yh[:, None]) * (m.conformal_l1_["Δŷ/ŷ"][q].predict(Z) + m.conformal_l2_["Δŷ/ŷ"][q])
        res["ct", cov] = score(iv[:, 0], iv[:, 1], yt, cov) + [tf + tp]
        res["ct_abs", cov] = score(a[:, 0], a[:, 1], yt, cov); res["ct_rel", cov] = score(r[:, 0], r[:, 1], yt, cov)
        res["ct_crossed", cov] = [np.mean(iv[:, 0] > iv[:, 1])]
    res["l2"] = len(m.ŷ_calib_l2_)
    # pyconformal on the same rows: 70/30 split, calibration size as in conformal-tights
    nc = min(int(0.3 * n), 1440); pp = np.random.default_rng(seed + 1).permutation(n); ca, fi = pp[:nc], pp[nc:]
    t0 = time.time(); base = XGBRegressor(objective="reg:absoluteerror", n_jobs=1, random_state=seed).fit(Xa[fi], ya[fi]); tb = time.time() - t0
    for cov in COV:
        t0 = time.time(); s = SplitConformalRegressor(alpha=1 - cov); s.calibrate(y=ya[ca], y_pred=base.predict(Xa[ca]))
        iv = np.asarray(s.predict_interval(y_pred=base.predict(Xt))); res["pc_split", cov] = score(iv[:, 0], iv[:, 1], yt, cov) + [tb + time.time() - t0]
        t0 = time.time(); ql = [(1 - cov) / 2, 1 - (1 - cov) / 2]
        qm = XGBRegressor(objective="reg:quantileerror", quantile_alpha=ql, n_jobs=1, random_state=seed).fit(Xa[fi], ya[fi])
        c = ConformalizedQuantileRegressor(alpha=1 - cov); c.calibrate(y=ya[ca], quantiles=qm.predict(Xa[ca]))
        iv = np.asarray(c.predict_interval(quantiles=qm.predict(Xt))); res["pc_cqr", cov] = score(iv[:, 0], iv[:, 1], yt, cov) + [time.time() - t0]
    return res
allres = {}
for n in (300, 1000, 5000):
    if len(y) < n + 140: continue
    t0 = time.time(); allres[n] = [trial(n, 7919 * n + s) for s in range(R)]
    c = lambda k: np.mean([r[k][0] for r in allres[n]])
    print(f"{name} n={n} L2={allres[n][0]['l2']} ct {c(('ct',0.9)):.3f}/{c(('ct',0.95)):.3f} split {c(('pc_split',0.9)):.3f}/{c(('pc_split',0.95)):.3f} cqr {c(('pc_cqr',0.9)):.3f}/{c(('pc_cqr',0.95)):.3f}  {time.time()-t0:.0f}s", flush=True)
    pickle.dump(allres, open(f"{out}/e4_{name}.pkl", "wb"))
