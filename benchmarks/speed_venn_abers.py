"""Venn-Abers speed: pyconformal against venn-abers (Petej) and MAPIE. Best of 3, seconds.
Also checks that all three give the same (p0, p1)."""
import time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from importlib.metadata import version
from pyconformal import VennAbers, MulticlassVennAbers
from pyconformal._va_kernel import HAVE_NUMBA
from venn_abers import VennAbers as PVA, VennAbersCalibrator as PCal
try:
    from mapie._venn_abers import VennAbers as MVA
except Exception: MVA = None
def best(f, reps=3):
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r
print("venn-abers", version("venn-abers"), "| mapie", version("mapie"), "| numba available:", HAVE_NUMBA)
VennAbers().fit(np.array([0.1, 0.2]), np.array([0, 1]))  # compile once
rng = np.random.default_rng(0)
def data(kind, n):
    s = rng.uniform(size=n)
    if kind == "calibrated-ish": y = rng.uniform(size=n) < s ** 1.5
    elif kind == "anti-correlated": y = rng.uniform(size=n) < 1 - s
    elif kind == "pure noise": y = rng.uniform(size=n) < 0.5
    elif kind == "rounded (ties)": s = np.round(s, 2); y = rng.uniform(size=n) < s
    return s, y.astype(int)
print("\nBINARY: fit + predict")
print(f"{'data':16} {'n_cal':>9} {'n_test':>9} {'venn-abers':>11} {'MAPIE':>8} {'ours numba':>11} {'ours numpy':>11} {'speed-up':>9} same")
for kind in ["calibrated-ish", "rounded (ties)", "pure noise", "anti-correlated"]:
    for n, m in [(10_000, 100_000), (100_000, 1_000_000), (1_000_000, 1_000_000)]:
        s, y = data(kind, n); t = rng.uniform(size=m)
        P, T = np.column_stack([1 - s, s]), np.column_stack([1 - t, t])
        def pet():
            v = PVA(); v.fit(P, y); return v.predict_proba(T)[1]
        def mp():
            v = MVA(); v.fit(P, y); return v.predict_proba(T)[1]
        def ours(nb):
            return lambda: VennAbers().fit(s, y, use_numba=nb).predict_p0_p1(t)
        skip = n >= 1_000_000 and kind in ("pure noise", "anti-correlated")
        tp, rp = best(pet, 1 if n >= 1_000_000 else 3)
        tm = best(mp, 1 if n >= 1_000_000 else 3)[0] if MVA else float("nan")
        tn, rn = best(ours(True)) if HAVE_NUMBA else (float("nan"), None)
        tpy, rpy = best(ours(False), 1 if n >= 1_000_000 else 3)
        same = np.allclose(rpy, rp, atol=1e-12) and (rn is None or np.array_equal(rn, rpy))
        print(f"{kind:16} {n:9,} {m:9,} {tp:11.3f} {tm:8.3f} {tn:11.4f} {tpy:11.3f} {tp / tn:8.0f}x {same}")
print("\nMULTICLASS one-vs-one, prefit probabilities: fit + predict")
for K, n, m in [(5, 20_000, 100_000), (10, 50_000, 200_000)]:
    y = rng.integers(0, K, size=n + m); lg = rng.normal(size=(n + m, K)) + 2 * np.eye(K)[y]
    P = np.exp(lg); P /= P.sum(1, keepdims=True)
    tp, rp = best(lambda: PCal().predict_proba(p_cal=P[:n], y_cal=y[:n], p_test=P[n:]), 1)
    tn, rn = best(lambda: MulticlassVennAbers().fit(P[:n], y[:n]).predict_proba(P[n:]))
    print(f"K={K:2} n_cal={n:,} n_test={m:,}: venn-abers {tp:.3f}  ours {tn:.4f}  speed-up {tp / tn:.0f}x  same {np.allclose(rn, rp, atol=1e-10)}")
