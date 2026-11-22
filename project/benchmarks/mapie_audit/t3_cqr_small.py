"""CQR with small calibration sets. The quantile 'models' are fixed (the true 5% and 95% quantiles of N(0,1)),
so calibration and test scores are exchangeable. Coverage must be at least the level, or the interval infinite."""
import sys, numpy as np
from common import *
from mapie.regression import ConformalizedQuantileRegressor
R = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NT = 200
print("level symmetric allow_inf n_cal  coverage (se)     share infinite  verdict")
for level in (0.9, 0.8):
    z = {0.9: 1.6449, 0.8: 1.2816}[level]
    for sym in (False, True):
        for inf in (False, True):
            for n_cal in (5, 8, 10, 15, 19, 20, 30, 39, 40, 60, 100):
                cov, infs, err = [], [], None
                for r in range(R):
                    rng = np.random.default_rng(1000 * r + n_cal)
                    X = np.zeros((n_cal + NT, 1)); y = rng.normal(size=n_cal + NT)
                    try:
                        m = ConformalizedQuantileRegressor([Const(-z), Const(z), Const(0.0)], confidence_level=level, prefit=True)
                        m.conformalize(X[:n_cal], y[:n_cal])
                        iv = m.predict_interval(X[n_cal:], symmetric_correction=sym, allow_infinite_bounds=inf)[1][:, :, 0]
                        cov.append(((iv[:, 0] <= y[n_cal:]) & (y[n_cal:] <= iv[:, 1])).mean()); infs.append(np.isinf(iv).any(1).mean())
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:70]}"; break
                if err: print(f"{level}  {str(sym):5s}     {str(inf):5s}   {n_cal:4d}   REFUSES: {err}", flush=True); continue
                m_, se = summary(cov)
                print(f"{level}  {str(sym):5s}     {str(inf):5s}   {n_cal:4d}   {m_:.4f} ({se:.4f})  {np.mean(infs):.2f}            {verdict(m_, se, level)}", flush=True)
