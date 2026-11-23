"""W-TQA on the Smart-Grid Smart-City household panel, following Tu and Giesecke (arXiv:2605.17705) as far as specified."""
import sys, time, numpy as np, pandas as pd
from pyconformal import WeightedTQA
REPS = int(sys.argv[1]); P_REVEAL = 0.2
Yd = pd.read_pickle("Y.pkl"); H = pd.read_csv("household.csv", low_memory=False).drop_duplicates("CUSTOMER_KEY").set_index("CUSTOMER_KEY")
end = pd.Timestamp("2013-03-17 23:30"); b0 = Yd.index.get_loc(end) + 1 - 3984; Yd = Yd.iloc[b0:b0 + 3984 + 600]; b, T = 3984, 600
flags = [c for c in H.columns if c.startswith("HAS_")] + ["IS_HOME_DURING_DAYTIME"]
num = ["NUM_REFRIGERATORS", "NUM_ROOMS_HEATED"]; cat = ["ASSRTD_CLIMATE_ZONE_CD", "ASSRTD_DWELLING_TYPE_CD", "ASSRTD_GAS_USAGE_GROUP_CD", "ASSRTD_ELECTRICITY_USE_GRP_CD", "DWELLING_TYPE_CD", "AIRCON_TYPE_CD"]
Hs = H.reindex(Yd.columns); ok = Hs[flags + num].notna().all(axis=1) & Yd.notna().all().values
Yd = Yd.loc[:, ok.values]; Hs = Hs.loc[ok.values]; n_all = Yd.shape[1]
A = pd.concat([(Hs[flags] == "Y").astype(float), Hs[num].apply(pd.to_numeric, errors="coerce").fillna(0.0), pd.get_dummies(Hs[cat].astype(str), dtype=float)], axis=1)
A = A.loc[:, A.std() > 0].values; zone = Hs["ASSRTD_CLIMATE_ZONE_CD"].astype(str).values
Y = np.log1p(Yd.values.astype(float)); idx = Yd.index
print(f"eligible households {n_all} (paper: 400), attribute columns {A.shape[1]} (paper: 31); burn-in {idx[0]} to {idx[b - 1]}; conformal {idx[b]} to {idx[-1]}")
hod = (idx.hour * 2 + idx.minute // 30).values; dow = idx.dayofweek.values; mon = idx.month.values
calv = np.c_[np.sin(2 * np.pi * hod / 48), np.cos(2 * np.pi * hod / 48), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12), (dow >= 5).astype(float)]
bmean, bsd, bzero = Y[:b].mean(0), Y[:b].std(0), (Y[:b] == 0).mean(0)
slot = np.vstack([Y[:b][hod[:b] == s].mean(0) for s in range(48)])[hod]           # burn-in mean at the round's half-hour slot
def run(rep):
    rng = np.random.default_rng(rep); perm = rng.permutation(n_all)[:400]; ci, ti = perm[:280], perm[280:]
    s = Y[:, ci].sum(1); peer = np.repeat((s / 280)[:, None], n_all, 1); peer[:, ci] = (s[:, None] - Y[:, ci]) / 279
    pz = np.empty_like(peer)
    for z in np.unique(zone):
        cz = ci[zone[ci] == z]; m = zone == z
        if len(cz) < 2: pz[:, m] = peer[:, m]; continue
        sz = Y[:, cz].sum(1); pz[:, m] = (sz / len(cz))[:, None]; pz[:, cz] = (sz[:, None] - Y[:, cz]) / (len(cz) - 1)
    xc = peer[:b] - peer[:b].mean(0); beta = (xc * (Y[:b] - Y[:b].mean(0))).sum(0) / np.maximum((xc * xc).sum(0), 1e-12)
    def feats(ix):
        cols = [np.repeat(calv[:, None, :], len(ix), 1), np.repeat(A[ix][None], len(idx), 0), peer[:, ix, None], pz[:, ix, None], (beta[ix] * peer[:, ix])[:, :, None]]
        st = np.stack([bmean[ix], bsd[ix], bzero[ix], beta[ix]], 1); cols += [np.repeat(st[None], len(idx), 0), slot[:, ix, None]]
        return np.concatenate(cols, 2)
    Fc, Ft = feats(ci), feats(ti); d = Fc.shape[2]; sub = rng.choice(b, 600, replace=False)       # ridge on a sample of burn-in rounds
    Xtr = Fc[sub].reshape(-1, d); ytr = Y[sub][:, ci].reshape(-1); mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1; Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + np.eye(d), Z.T @ (ytr - ytr.mean())); pred = lambda F: ((F - mu) / sd) @ w + ytr.mean()
    Sc = np.abs(Y[b:, ci] - pred(Fc[b:])); St = np.abs(Y[b:, ti] - pred(Ft[b:]))
    na = A.shape[1]; prof = list(range(7, 7 + na)) + list(range(7 + na + 3, 7 + na + 7))             # household attributes, burn-in statistics, peer slope
    R = rng.random(T) < P_REVEAL; out = {}
    for name, h, g in (("Split CP", np.inf, 0.0), ("TQA-only", np.inf, 0.01), ("W-only", 0.6, 0.0), ("W-TQA", 0.6, 0.01)):
        r = WeightedTQA(alpha=0.1, bandwidth=h, gamma=g, finite=True).run(scores_cal=Sc, scores_target=St, X_cal=Fc[b:, :, prof], X_target=Ft[b:, :, prof], revealed=R)
        wd = 2 * r.thresholds; out[name] = (r.coverage, r.tail_coverage(0.1), wd.mean(), wd.std() / wd.mean(), r.fallback.mean())
    return out
t0 = time.time(); res = [run(r) for r in range(REPS)]; print(f"{REPS} replications, feedback probability {P_REVEAL}, finite=True, {time.time() - t0:.0f} s")
paper = {"Split CP": (0.902, 0.646, 1.0, 0.346), "TQA-only": (0.924, 0.836, 1.024, 0.489), "W-only": (0.933, 0.763, 1.297, 0.566), "W-TQA": (0.935, 0.858, 1.172, 0.621)}
print("  method     | avg cov (paper)      | tail cov (paper)     | rel width (paper) | width CoV (paper) | fallback rounds")
for name, p in paper.items():
    a = np.array([r[name] for r in res]); base = np.array([r["Split CP"][2] for r in res]); m, s = a.mean(0), a.std(0)
    print(f"  {name:10s} | {m[0]:.3f}+-{s[0]:.3f} ({p[0]:.3f}) | {m[1]:.3f}+-{s[1]:.3f} ({p[1]:.3f}) | {(a[:, 2] / base).mean():.3f} ({p[2]:.3f}) | {m[3]:.3f} ({p[3]:.3f}) | {m[4]:.4f}")
