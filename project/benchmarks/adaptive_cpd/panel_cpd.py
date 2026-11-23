"""Option 2: a predictive distribution for each target unit of a panel.
Base = the weighted distribution of the peers' signed errors in the same round (weights as in W-TQA, or equal).
Adaptive = that base passed through PITTracker, fed only on the rounds where the target's outcome is revealed.
Panels and pipelines are those of ../wtqa (M5 and SGSC); 19 levels 0.05..0.95; gamma 0.01, bandwidth 0.6, feedback on 20% of rounds."""
import sys, os, re, io, contextlib, time, numpy as np
from pyconformal import WeightedTQA
from adaptive_cpd import PITTracker, weighted_quantiles, pinball
levels = np.round(np.arange(0.05, 0.96, 0.05), 2)
def loader(panel):
    """Reuse the W-TQA benchmark script of the panel, returning signed errors instead of running the interval methods."""
    d = {"m5": ("/home/claude/wtqa/m5", "../wtqa/m5_wtqa.py"), "sgsc": ("/home/claude/wtqa/sgsc", "../wtqa/sgsc/sgsc_wtqa.py")}[panel]
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), d[1])).read()
    src = src.replace("np.abs(Y[b:b + T, ci] - Pc)", "(Y[b:b + T, ci] - Pc)").replace("np.abs(Y[b:b + T, ti] - Pt)", "(Y[b:b + T, ti] - Pt)")
    src = src.replace("np.abs(Y[b:, ci] - pred(Fc[b:]))", "(Y[b:, ci] - pred(Fc[b:]))").replace("np.abs(Y[b:, ti] - pred(Ft[b:]))", "(Y[b:, ti] - pred(Ft[b:]))")
    a = src.index("    R = rng.random(T) < P_REVEAL"); b_ = src.index("t0 = time.time()")
    src = src[:a] + "    R = rng.random(T) < P_REVEAL\n    return Sc, St, Fc[b:, :, prof], Ft[b:, :, prof], R\n"
    ns = {}; cwd = os.getcwd(); os.chdir(d[0]); argv = sys.argv; sys.argv = ["x", "0"]
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, d[1], "exec"), ns)
    os.chdir(cwd); sys.argv = argv; return ns["run"]
def one(run, rep, gamma=0.01, h=0.6):
    Rc, Rt, Xc, Xt, R = run(rep); T, N = Rc.shape; M = Rt.shape[1]; out = {}
    for wname, bw in (("equal weights", np.inf), ("similarity weights", h)):
        tq = WeightedTQA(bandwidth=bw).start(N, M, Xc.shape[2]); trk = PITTracker(levels, gamma, M, "sorted")
        Qb = np.empty((T, M, levels.size)); Qa = np.empty_like(Qb); lo = np.empty(T); hi = np.empty(T)
        for t in range(T):
            W = tq.weights(); order = np.argsort(Rc[t]); rs = Rc[t][order]; cum = np.cumsum(W[:, :N][:, order], axis=1); wt = W[:, N]
            Qb[t] = weighted_quantiles(rs, cum, wt, np.repeat(levels[None], M, 0)); Qa[t] = weighted_quantiles(rs, cum, wt, trk.thresholds())
            pit = 0.5 * wt + np.take_along_axis(np.c_[np.zeros(M), cum], (rs[None, :] <= Rt[t][:, None]).sum(1)[:, None], 1)[:, 0]
            trk.update(pit, R[t]); tq.update(Xc[t], Xt[t]); lo[t], hi[t] = rs[0], rs[-1]
        for aname, Q in (("fixed levels", Qb), ("adaptive levels", Qa)):
            hit = Rt[:, :, None] <= Q; cal_u = np.abs(hit.mean(0) - levels)                       # (M, K) calibration error of each unit at each level
            unit = cal_u.mean(1); worst = np.sort(unit)[-max(1, int(np.ceil(0.1 * M))):].mean()
            Qc = np.clip(Q, lo[:, None, None], hi[:, None, None]); pb = pinball(Qc, Rt, levels)   # infinite quantiles scored at the round's extreme peer error
            c90 = ((Rt >= Q[:, :, 0]) & (Rt <= Q[:, :, -1])).mean(0); wd = Qc[:, :, -1] - Qc[:, :, 0]
            out[(wname, aname)] = (np.abs(hit.mean((0, 1)) - levels).max(), unit.mean(), worst, pb.mean(), np.sort(pb.mean(0))[-max(1, int(np.ceil(0.1 * M))):].mean(),
                                   c90.mean(), np.sort(c90)[:max(1, int(np.ceil(0.1 * M)))].mean(), wd.mean(), np.isinf(Q).mean())
    return out
if __name__ == "__main__":
    panel, reps = sys.argv[1], int(sys.argv[2]); run = loader(panel); t0 = time.time(); res = [one(run, r) for r in range(reps)]
    print(f"{panel.upper()}: {reps} random splits, {time.time() - t0:.0f} s. Mean over splits.")
    print("  distribution                         | pooled worst-level cal. error | per-unit cal. error: mean | worst 10% of units | pinball: mean | worst 10% of units | central 90% coverage | worst 10% of units | 90% width | infinite quantiles")
    for k in res[0]:
        m = np.mean([r[k] for r in res], 0); print(f"  {k[0] + ', ' + k[1]:36s} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.4f} | {m[4]:.4f} | {m[5]:.3f} | {m[6]:.3f} | {m[7]:.3f} | {m[8]:.4f}")
