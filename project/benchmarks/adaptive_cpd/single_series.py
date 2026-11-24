"""Option 1: one series. Base = rolling conformal predictive system on the last W forecast errors.
Adaptive = the same base passed through PITTracker. Scored on per-level calibration and pinball loss."""
import sys, numpy as np, pandas as pd
from adaptive_cpd import PITTracker, rolling_cps, rolling_pit, pinball
levels = np.round(np.arange(0.05, 0.96, 0.05), 2); W = 200
def streams(seed, T=3000):
    g = np.random.default_rng(seed); z = g.normal(size=T + W); t = np.arange(T + W)
    return {"i.i.d. errors": z, "noise x5 from the middle": z * np.where(t < W + T // 2, 1, 5), "noise grows 0.1% a step": z * 1.001 ** t,
            "forecast bias drifts (error mean +0.002 a step)": z + 0.002 * t, "random-walk error": np.cumsum(z) * 0.1,
            "regime switches every 250 steps (scale 1 / 4)": z * np.where((t // 250) % 2 == 0, 1, 4), "skew flips at the middle": np.where(t < W + T // 2, 1, -1) * (g.exponential(size=T + W) - 1)}
def real():
    out = {}
    try:
        Y = pd.read_pickle("/home/claude/wtqa/m5/Y.pkl").sum(axis=1).values; out["M5 CA_1 FOODS_3 daily total, seasonal-naive forecast (lag 7)"] = Y[7:] - Y[:-7]
        S = pd.read_pickle("/home/claude/wtqa/sgsc/Y.pkl").iloc[:, :500].mean(axis=1).values; out["SGSC mean household load, half-hourly, forecast = same slot yesterday"] = S[48:] - S[:-48]
    except Exception as e: print("real series not available:", e)
    return out
def run(e, gamma, mode):
    T = e.size - W; tr = PITTracker(levels, gamma, 1, mode); Qb = np.empty((T, levels.size)); Qa = np.empty_like(Qb); cross = 0
    for i in range(T):
        win = e[i:i + W]; Qb[i] = rolling_cps(win, levels); th = tr.thresholds()[0]; cross += np.any(np.diff(tr.theta[0]) < 0)
        Qa[i] = rolling_cps(win, th); tr.update(rolling_pit(win, e[i + W]))
    y = e[W:]; return y, Qb, Qa, cross / T
def score(y, Q):
    cal = np.abs((y[:, None] <= Q).mean(0) - levels); fin = np.isfinite(Q)
    lo, hi = np.nanmin(np.where(fin, Q, np.nan)), np.nanmax(np.where(fin, Q, np.nan)); Qc = np.clip(Q, lo, hi)     # infinite quantiles scored at the widest finite one
    c90 = ((y >= Q[:, 0]) & (y <= Q[:, -1])).mean(); roll = pd.Series(((y >= Q[:, 0]) & (y <= Q[:, -1])).astype(float)).rolling(250).mean().min()
    return cal.max(), cal.mean(), pinball(Qc, y, levels).mean(), c90, roll, 1 - fin.mean()
if __name__ == "__main__":
    R = int(sys.argv[1]); gammas = (0.005, 0.02, 0.05)
    print(f"19 levels 0.05..0.95, window {W}. Columns: worst calibration error over levels | mean calibration error | pinball loss | coverage of the central 90% interval | lowest coverage in any 250-step stretch | share of infinite quantiles")
    print(f"Bound on the calibration error of any level for any sequence, T = 3000: " + ", ".join(f"gamma {g}: {(1 + g) / (g * 3000):.3f}" for g in gammas))
    names = list(streams(0))
    for name in names:
        print(f"\n{name}")
        rows = {}
        for seed in range(R):
            e = streams(seed)[name]
            for g in gammas:
                y, Qb, Qa, cr = run(e, g, "sorted"); rows.setdefault("base (rolling CPS)", []).append(score(y, Qb) + (0.0,)); rows.setdefault(f"adaptive, gamma {g}", []).append(score(y, Qa) + (cr,))
            y, Qb, Qi, cr = run(e, 0.02, "independent"); rows.setdefault("adaptive, gamma 0.02, not sorted", []).append(score(y, Qi) + (cr,))
        for k, v in rows.items():
            m = np.mean(v, 0); print(f"  {k:34s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.4f} | {m[3]:.3f} | {m[4]:.3f} | {m[5]:.4f}" + (f" | steps with crossed thresholds before sorting {m[6]:.3f}" if m[6] else ""))
    for name, e in real().items():
        print(f"\n{name} ({e.size - W} steps)")
        y, Qb, _, _ = run(e, 0.02, "sorted"); m = score(y, Qb); print(f"  {'base (rolling CPS)':34s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.4f} | {m[3]:.3f} | {m[4]:.3f} | {m[5]:.4f}")
        for g in gammas:
            y, _, Qa, cr = run(e, g, "sorted"); m = score(y, Qa); print(f"  {f'adaptive, gamma {g}':34s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.4f} | {m[3]:.3f} | {m[4]:.3f} | {m[5]:.4f}")
