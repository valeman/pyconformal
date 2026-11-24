"""Panel: every recalibrator on the same base (the weighted distribution of the peers' signed errors in the round),
one recalibrator per target, fed only on the rounds where the target's outcome is revealed (20%). 19 levels."""
import sys, time, numpy as np
from pyconformal import WeightedTQA
from adaptive_cpd import PITTracker, weighted_quantiles, pinball
from comparators import LazyPIT, EmpiricalPIT, Jumper, MultiQT
from panel_cpd import loader, levels
def one(run, rep, gamma=0.01, h=0.6):
    Rc, Rt, Xc, Xt, R = run(rep); T, N = Rc.shape; M = Rt.shape[1]; K = levels.size; out = {}
    for wname, bw in (("equal weights", np.inf), ("similarity weights", h)):
        tq = WeightedTQA(bandwidth=bw).start(N, M, Xc.shape[2])
        pitm = {"PIT tracker, one per level": PITTracker(levels, gamma, M, "independent"), "PIT tracker, lazy isotonic": LazyPIT(levels, gamma, M),
                "Online recalibration (empirical PIT)": EmpiricalPIT(levels, M), "Betting enhancement (Vovk)": Jumper(levels, M)}
        mq = MultiQT(levels, M); names = ["Base", *pitm, "MultiQT (outcome scale)"]; Q = {k: np.empty((T, M, K)) for k in names}; lo = np.empty(T); hi = np.empty(T)
        for t in range(T):
            W = tq.weights(); order = np.argsort(Rc[t]); rs = Rc[t][order]; cum = np.cumsum(W[:, :N][:, order], axis=1); wt = W[:, N]; lo[t], hi[t] = rs[0], rs[-1]
            base = weighted_quantiles(rs, cum, wt, np.repeat(levels[None], M, 0)); Q["Base"][t] = base
            pit = 0.5 * wt + np.take_along_axis(np.c_[np.zeros(M), cum], (rs[None, :] <= Rt[t][:, None]).sum(1)[:, None], 1)[:, 0]
            for k, m in pitm.items(): Q[k][t] = weighted_quantiles(rs, cum, wt, m.thresholds()); m.update(pit, R[t])
            bc = np.clip(base, rs[0], rs[-1]); Q["MultiQT (outcome scale)"][t] = mq.forecast(bc); mq.update(bc, Rt[t], R[t]); tq.update(Xc[t], Xt[t])
        for k, q in Q.items():
            hit = Rt[:, :, None] <= q; unit = np.abs(hit.mean(0) - levels).mean(1); n10 = max(1, int(np.ceil(0.1 * M)))
            qc = np.clip(q, lo[:, None, None], hi[:, None, None]); pb = pinball(qc, Rt, levels).mean(0); c90 = ((Rt >= q[:, :, 0]) & (Rt <= q[:, :, -1])).mean(0)
            out[(wname, k)] = (unit.mean(), np.sort(unit)[-n10:].mean(), pb.mean(), np.sort(pb)[-n10:].mean(), c90.mean(), np.sort(c90)[:n10].mean(), (qc[:, :, -1] - qc[:, :, 0]).mean(), np.isinf(q).mean(), np.mean(np.diff(q, axis=2) < -1e-12))
    return out
if __name__ == "__main__":
    panel, reps = sys.argv[1], int(sys.argv[2]); run = loader(panel); t0 = time.time(); res = [one(run, r) for r in range(reps)]
    print(f"{panel.upper()}: {reps} random splits, {time.time() - t0:.0f} s. Mean over splits.")
    print("  weights, recalibrator | per-unit calibration error: mean | worst 10% of units | pinball: mean | worst 10% | central 90% coverage | worst 10% | 90% width | infinite quantiles | crossed pairs")
    for k in res[0]:
        m = np.mean([r[k] for r in res], 0); print(f"  {k[0] + ', ' + k[1]:58s} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.4f} | {m[3]:.4f} | {m[4]:.3f} | {m[5]:.3f} | {m[6]:.3f} | {m[7]:.4f} | {m[8]:.4f}")
