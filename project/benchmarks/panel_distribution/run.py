"""PanelPredictiveSystem on the M5 and SGSC panels of ../wtqa (same pipelines, splits and feedback).
Usage: python run.py m5|sgsc n_splits [feedback_probability]"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adaptive_cpd"))
from panel_cpd import loader
from pyconformal import PanelPredictiveSystem
def metrics(Q, Y, levels):
    hit = Y[:, :, None] <= Q; unit = np.abs(hit.mean(0) - levels).mean(1); n10 = max(1, int(np.ceil(0.1 * Y.shape[1])))
    d = Y[:, :, None] - Q; pb = np.maximum(levels * d, (levels - 1) * d).mean((0, 2)); c90 = ((Y >= Q[:, :, 0]) & (Y <= Q[:, :, -1])).mean(0)
    return unit.mean(), np.sort(unit)[-n10:].mean(), pb.mean(), np.sort(pb)[-n10:].mean(), c90.mean(), np.sort(c90)[:n10].mean(), (Q[:, :, -1] - Q[:, :, 0]).mean(), np.abs(hit.mean((0, 1)) - levels).max()
if __name__ == "__main__":
    panel, reps = sys.argv[1], int(sys.argv[2]); p = float(sys.argv[3]) if len(sys.argv) > 3 else None; run = loader(panel); out = {}; t0 = time.time(); secs = 0.0
    for r in range(reps):
        Rc, Rt, Xc, Xt, R = run(r); Z = np.zeros_like
        if p is not None: R = np.random.default_rng(1000 + r).random(Rc.shape[0]) < p
        for name, kw in (("equal weights", dict(bandwidth=np.inf)), ("similarity weights", dict()), ("similarity weights, step floor 0.02", dict(lr_floor=0.02))):
            t1 = time.time(); res = PanelPredictiveSystem(**kw).run(y_pred_cal=Z(Rc), y_cal=Rc, y_pred_target=Z(Rt), y_target=Rt, X_cal=Xc, X_target=Xt, revealed=R); secs += time.time() - t1
            if not name.endswith("0.02"): out.setdefault((name, "base only (no adjustment)"), []).append(metrics(res.base_quantiles, Rt, res.levels))
            out.setdefault((name, "PanelPredictiveSystem"), []).append(metrics(res.quantiles, Rt, res.levels))
    print(f"{panel.upper()}: {reps} splits, feedback on {'20' if p is None else round(100 * p)}% of rounds; {Rc.shape[0]} rounds, {Rc.shape[1]} calibration units, {Rt.shape[1]} targets, 19 levels; {secs / (3 * reps):.2f} s per run of the method")
    print("  variant | per-unit calibration error | worst 10% of units | pinball | worst 10% | central 90% coverage | worst 10% | 90% width | pooled worst-level error")
    for k, v in out.items():
        m = np.mean(v, 0); print(f"  {k[0] + ': ' + k[1]:62s} | {m[0]:.3f} | {m[1]:.3f} | {m[2]:.4f} | {m[3]:.4f} | {m[4]:.3f} | {m[5]:.3f} | {m[6]:.3f} | {m[7]:.3f}")
