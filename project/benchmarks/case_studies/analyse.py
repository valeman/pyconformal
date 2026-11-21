import numpy as np, warnings, time
warnings.filterwarnings("ignore")
from pyconformal import MultiStepSplitConformal, AcMCP

def load(series, model):
    z = np.load(f"sf_{series}_{model}.npz")
    y, origins, mean, H = z["y"], z["origins"], z["mean"], int(z["H"])
    N = len(origins); t0 = origins[0]; T = N                  # targets t0 .. t0+N-1
    yt = y[t0:t0 + T]
    def by_target(A):                                          # A[i, h-1] -> M[t, h-1], target t = i + h - 1
        M = np.full((T, H), np.nan)
        for h in range(1, H + 1):
            M[h - 1:, h - 1] = A[:N - h + 1, h - 1]
        return M
    return yt, by_target(mean), {nw: (by_target(z[f"lo{nw}"]), by_target(z[f"hi{nw}"])) for nw in (2, 10)}, H

def summarise(series, model, warm=100, verbose=True):
    yt, F, sf, H = load(series, model)
    res = {}
    t = time.time(); r = MultiStepSplitConformal(alpha=0.1, symmetric=True, min_calibration=30).run(y_pred=F, y=yt); t_ms = time.time() - t
    res["pyconformal multi-step split"] = (r.intervals[..., 0], r.intervals[..., 1])
    r = MultiStepSplitConformal(alpha=0.1, symmetric=True, min_calibration=30, window=200).run(y_pred=F, y=yt)
    res["pyconformal multi-step split, window 200"] = (r.intervals[..., 0], r.intervals[..., 1])
    t = time.time(); r = AcMCP(alpha=0.1, min_calibration=30, scorecast=False).run(y_pred=F, y=yt); t_ac = time.time() - t
    res["pyconformal AcMCP"] = (r.intervals[..., 0], r.intervals[..., 1])
    res["statsforecast default (2 windows)"] = sf[2]
    res["statsforecast 10 windows"] = sf[10]
    ok = np.zeros_like(F, bool); ok[warm + H:] = True
    ok &= ~np.isnan(F)
    for k, (lo, hi) in res.items():
        ok &= ~np.isnan(lo)
    out = {}
    for k, (lo, hi) in res.items():
        cov = ((yt[:, None] >= lo) & (yt[:, None] <= hi))
        covh = np.array([cov[ok[:, h], h].mean() for h in range(H)])
        w = (hi - lo)
        fin = np.isfinite(w) & ok
        out[k] = dict(cov=cov[ok].mean(), covh=covh, width=np.median(w[fin]), inf=(~np.isfinite(w) & ok).sum())
    if verbose:
        print(f"\n== {series} / {model}: {ok[:, 0].sum()} targets per horizon, H={H}; MSCP {t_ms*1e3:.0f} ms, AcMCP {t_ac*1e3:.0f} ms")
        for k, d in out.items():
            print(f"  {k:42s} cov {d['cov']:.3f}  min-h {d['covh'].min():.3f}  max-h {d['covh'].max():.3f}  median width {d['width']:.1f}  inf {d['inf']}")
    return yt, F, res, ok, out, H

if __name__ == "__main__":
    import sys, os
    for s, m in [("births", "snaive"), ("river", "snaive"), ("power", "snaive"), ("births", "ets"), ("river", "ets"), ("power", "theta")]:
        if os.path.exists(f"sf_{s}_{m}.npz"):
            summarise(s, m)
