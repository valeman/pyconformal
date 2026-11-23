"""W-TQA on the M5 CA_1 / FOODS_3 panel, following Section 6 and Appendix C of Tu and Giesecke (arXiv:2605.17705)
as far as the paper specifies it. Differences are listed in the README beside this file."""
import sys, time, numpy as np, pandas as pd
from pyconformal import WeightedTQA
REPS = int(sys.argv[1]); P_REVEAL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
Y = pd.read_pickle("Y.pkl"); P = pd.read_pickle("P.pkl"); snap = pd.read_pickle("snap.pkl")
burn_end = pd.Timestamp("2013-12-31"); days = Y.index; b = int(np.searchsorted(days, burn_end, side="right")); T = 600
first = P.notna().values.argmax(axis=0)
keep = (first <= b - 180) & P.iloc[b:b + T].notna().all().values          # at least 180 burn-in days on sale, on sale throughout
Y = np.log1p(Y.values[:, keep]); P = P.values[:, keep]; first = first[keep]; n_all = Y.shape[1]
print(f"items kept {n_all} (paper: 648); burn-in days {b}; conformal days {T}: {days[b].date()} to {days[b + T - 1].date()}")
onsale = ~np.isnan(P)
dow = days.dayofweek.values; mon = days.month.values
cal = np.c_[np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12), snap.values]
logp = np.log(P); chg = lambda k: np.vstack([np.zeros((k, n_all)), logp[k:] - logp[:-k]])
c7, c28 = np.nan_to_num(chg(7)), np.nan_to_num(chg(28))
med = np.nanmedian(P[:b], axis=0); rel = P / med; age = (np.arange(len(days))[:, None] - first[None, :]) / 365.0
Yb = np.where(onsale[:b], Y[:b], np.nan); bmean = np.nanmean(Yb, 0); bsd = np.nanstd(Yb, 0); bzero = np.nanmean(Yb == 0, 0)
rows = slice(0, b + T)
def run(rep):
    rng = np.random.default_rng(rep); perm = rng.permutation(n_all)[:648] if n_all >= 648 else rng.permutation(n_all)
    ncal = int(round(len(perm) * 454 / 648)); ci, ti = perm[:ncal], perm[ncal:]
    Yc = np.where(onsale[:, ci], Y[:, ci], np.nan); s = np.nansum(Yc, 1); cnt = (~np.isnan(Yc)).sum(1)
    peer = np.empty((len(days), n_all)); peer[:] = (s / np.maximum(cnt, 1))[:, None]
    own = np.nan_to_num(Yc); peer[:, ci] = (s[:, None] - own) / np.maximum(cnt[:, None] - (~np.isnan(Yc)), 1)   # leave the unit itself out
    beta = np.empty(n_all)
    for i in range(n_all):
        m = onsale[:b, i]; x = peer[:b, i][m]; y = Y[:b, i][m]; xc = x - x.mean(); beta[i] = (xc @ (y - y.mean())) / max(xc @ xc, 1e-12)
    def feats(idx):
        n = len(idx); F = np.empty((b + T, n, 16))
        F[:, :, :5] = cal[rows, None, :]; F[:, :, 5] = c7[rows][:, idx]; F[:, :, 6] = c28[rows][:, idx]
        F[:, :, 7] = logp[rows][:, idx]; F[:, :, 8] = rel[rows][:, idx]; F[:, :, 9] = age[rows][:, idx]
        F[:, :, 10] = peer[rows][:, idx]; F[:, :, 11] = beta[idx] * peer[rows][:, idx]
        F[:, :, 12] = bmean[idx]; F[:, :, 13] = bsd[idx]; F[:, :, 14] = bzero[idx]; F[:, :, 15] = beta[idx]
        return F
    Fc, Ft = feats(ci), feats(ti)
    mask = onsale[:b, ci]; Xtr = Fc[:b][mask]; ytr = Y[:b, ci][mask]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1; Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(16), Z.T @ (ytr - ytr.mean())); pred = lambda F: ((F - mu) / sd) @ w + ytr.mean()
    Pc, Pt = pred(Fc[b:]), pred(Ft[b:]); Sc = np.abs(Y[b:b + T, ci] - Pc); St = np.abs(Y[b:b + T, ti] - Pt)
    prof = [7, 8, 9, 12, 13, 14, 15]                                      # covariates that describe the unit itself
    R = rng.random(T) < P_REVEAL
    out = {}
    for name, h, g in (("Split CP", np.inf, 0.0), ("TQA-only", np.inf, 0.01), ("W-only", 0.6, 0.0), ("W-TQA", 0.6, 0.01)):
        for fin in (True, False):
            r = WeightedTQA(alpha=0.1, bandwidth=h, gamma=g, finite=fin).run(scores_cal=Sc, scores_target=St, X_cal=Fc[b:, :, prof], X_target=Ft[b:, :, prof], revealed=R)
            wdt = 2 * r.thresholds; fw = wdt[np.isfinite(wdt)]
            out[(name, fin)] = (r.coverage, r.tail_coverage(0.1), fw.mean(), fw.std() / fw.mean(), np.isinf(wdt).mean(), r.fallback.mean())
    return out
t0 = time.time(); res = [run(r) for r in range(REPS)]; print(f"{REPS} replications, feedback probability {P_REVEAL}, {time.time() - t0:.0f} s")
paper = {"Split CP": (0.904, 0.579, 1.0, 0.076), "TQA-only": (0.931, 0.827, 0.986, 0.233), "W-only": (0.913, 0.697, 0.991, 0.332), "W-TQA": (0.925, 0.847, 0.963, 0.358)}
for fin in (True, False):
    print(f"\nfinite={fin}" + ("  (the paper's Algorithm 1)" if fin else "  (pyconformal default: infinite sets kept; widths over finite sets only)"))
    print("  method     | avg cov (paper)  | tail cov (paper) | rel width (paper) | width CoV (paper) | infinite sets | fallback rounds")
    for name in paper:
        a = np.array([r[(name, fin)] for r in res]); base = np.array([r[("Split CP", fin)][2] for r in res]); m = a.mean(0); s = a.std(0); p = paper[name]
        print(f"  {name:10s} | {m[0]:.3f}+-{s[0]:.3f} ({p[0]:.3f}) | {m[1]:.3f}+-{s[1]:.3f} ({p[1]:.3f}) | {(a[:, 2] / base).mean():.3f} ({p[2]:.3f}) | {m[3]:.3f} ({p[3]:.3f}) | {m[4]:.4f} | {m[5]:.4f}")
