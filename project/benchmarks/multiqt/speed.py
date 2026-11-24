import time, numpy as np, cProfile, pstats
src = open("multiQT/example.py").read(); a = src.index("def isotonic_regression"); b = src.index('if __name__ == "__main__"')
ns = {"np": np}; exec("from sklearn.isotonic import IsotonicRegression\n" + src[a:b], ns); theirs = ns["multiQT"]
from pyconformal.multiqt import MultiQT
rng = np.random.default_rng(0); levels = np.round(np.arange(0.05, 0.96, 0.05), 2)
MultiQT(levels).run(np.zeros((5, 19)), np.zeros(5))
for T in (2000, 20000):
    Y = np.cumsum(rng.normal(size=T)) * 0.2; base = np.r_[0, Y[:-1]][:, None] + np.quantile(rng.normal(size=500), levels)[None]
    t = time.perf_counter(); q2 = MultiQT(levels).run(base, Y).quantiles[:, 0]; a_ = time.perf_counter() - t
    n = min(T, 2000); t = time.perf_counter(); q1 = theirs(Y[:n], levels, base_forecasts=base[:n].T.copy()); b_ = (time.perf_counter() - t) * T / n
    print(f"1 series, {T} rounds, 19 levels: pyconformal {a_:.3f} s; authors' example code {b_:.1f} s ({'measured' if n == T else 'extrapolated from 2,000 rounds'}); ratio {b_ / a_:.0f}x; max difference {np.abs(q1.T - q2[:n]).max():.1e}", flush=True)
T, n = 600, 500; Y = np.cumsum(rng.normal(size=(T, n)), 0) * 0.2; base = np.r_[np.zeros((1, n)), Y[:-1]][:, :, None] + np.quantile(rng.normal(size=500), levels)[None, None]
t = time.perf_counter(); r = MultiQT(levels).run(base, Y); a_ = time.perf_counter() - t; print(f"{n} series side by side, {T} rounds: pyconformal {a_:.2f} s ({a_ / n * 1000:.2f} ms a series)")
pr = cProfile.Profile(); pr.enable(); MultiQT(levels).run(base, Y); pr.disable(); pstats.Stats(pr).sort_stats("tottime").print_stats(5)
