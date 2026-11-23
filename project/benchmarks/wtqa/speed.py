import time, sys, numpy as np, cProfile, pstats
sys.path.insert(0, "tests")
from pyconformal import WeightedTQA
from test_panel import reference_wtqa
rng = np.random.default_rng(0)
def panel(T, N, M, d):
    prof = rng.normal(size=(N + M, d)); X = prof[None] + 0.5 * rng.normal(size=(T, N + M, d)); S = np.abs(rng.normal(size=(T, N + M))) * (0.5 + np.abs(prof[:, 0]))
    return S[:, :N], S[:, N:], X[:, :N], X[:, N:]
for T, N, M, d in ((600, 450, 100, 10), (600, 450, 600, 10), (2000, 2000, 500, 20)):
    S, St, Xc, Xt = panel(T, N, M, d); R = rng.random((T, M)) < 0.2
    t = time.perf_counter(); res = WeightedTQA().run(scores_cal=S, scores_target=St, X_cal=Xc, X_target=Xt, revealed=R); a = time.perf_counter() - t
    t = time.perf_counter(); ref = reference_wtqa(S[:20], St[:20, :2], Xc[:20], Xt[:20, :2], R[:20, :2], 0.1, 0.6, 0.01, False); b = (time.perf_counter() - t) / (20 * 2) * T * M
    same = np.array_equal(res.thresholds[:20, :2], ref)
    print(f"T={T} N={N} M={M} d={d}: pyconformal {a:.2f} s; line-by-line reference (extrapolated from 40 target-rounds) {b:.0f} s; ratio {b / a:.0f}x; same thresholds on the checked part: {same}", flush=True)
S, St, Xc, Xt = panel(600, 450, 600, 10)
pr = cProfile.Profile(); pr.enable(); WeightedTQA().run(scores_cal=S, scores_target=St, X_cal=Xc, X_target=Xt); pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(6)
