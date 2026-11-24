"""One series: every recalibrator on the same rolling conformal predictive system (window 200), 19 levels."""
import sys, numpy as np, pandas as pd
from adaptive_cpd import PITTracker, rolling_cps, rolling_pit, pinball
from comparators import LazyPIT, EmpiricalPIT, Jumper, MultiQT
from single_series import streams, real, levels, W
def run(e, gamma=0.02):
    T = e.size - W; pitm = {"PIT tracker, one per level (ACI at each level)": PITTracker(levels, gamma, 1, "independent"), "PIT tracker, sorted (first prototype)": PITTracker(levels, gamma, 1, "sorted"),
        "PIT tracker, lazy isotonic (MultiQT rule on the PIT scale)": LazyPIT(levels, gamma, 1), "Online recalibration (empirical PIT)": EmpiricalPIT(levels, 1), "Betting enhancement (Vovk, Simple Jumper)": Jumper(levels, 1)}
    mq = MultiQT(levels, 1); Q = {k: np.empty((T, levels.size)) for k in ["Base (rolling CPS)", *pitm, "MultiQT (outcome scale, as published)"]}; cross = 0
    for i in range(T):
        win = e[i:i + W]; base = rolling_cps(win, levels); Q["Base (rolling CPS)"][i] = base; p = rolling_pit(win, e[i + W])
        for k, m in pitm.items():
            th = m.thresholds()[0]; Q[k][i] = rolling_cps(win, th); cross += (k.startswith("PIT tracker, one")) and bool(np.any(np.diff(th) < 0)); m.update(p)
        Q["MultiQT (outcome scale, as published)"][i] = mq.forecast(base[None])[0]; mq.update(base[None], e[i + W:i + W + 1])
    return e[W:], Q, cross / T
def score(y, Q):
    cal = np.abs((y[:, None] <= Q).mean(0) - levels); fin = np.isfinite(Q); lo, hi = np.nanmin(np.where(fin, Q, np.nan)), np.nanmax(np.where(fin, Q, np.nan)); Qc = np.clip(Q, lo, hi)
    roll = pd.DataFrame((y[:, None] <= Q).astype(float)).rolling(250).mean().sub(levels).abs().max().max()
    return cal.max(), cal.mean(), roll, pinball(Qc, y, levels).mean(), 1 - fin.mean(), float(np.mean(np.diff(Q, axis=1) < -1e-12))
if __name__ == "__main__":
    R = int(sys.argv[1]); print("Columns: worst-level calibration error | mean over levels | worst error of any level in any 250-step stretch | pinball loss | infinite quantiles | crossed quantile pairs")
    cases = [(n, [streams(s)[n] for s in range(R)]) for n in streams(0)] + [(n, [e]) for n, e in real().items()]
    for name, es in cases:
        print(f"\n{name}"); acc = {}
        for e in es:
            y, Q, cr = run(e)
            for k, q in Q.items(): acc.setdefault(k, []).append(score(y, q))
        for k, v in acc.items():
            m = np.mean(v, 0); print(f"  {k:58s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} | {m[3]:.4f} | {m[4]:.4f} | {m[5]:.4f}")
