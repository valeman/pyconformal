"""CIR and CIR+ (Guo, Luo and Zhou, AAAI 2026): pyconformal against the authors' code.

Usage: python speed_cir.py /path/to/clone/of/github.com/orince/CIR
The authors' module imports torch and torchcp. Both packages get the same
quantile predictions (101 quantiles, so 100 interquantile intervals).
"""
import importlib.util, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pyconformal as pc

spec = importlib.util.spec_from_file_location("cir_authors", sys.argv[1].rstrip("/") + "/chr/ours.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
class Stored:  # a "model" whose predictions are the stored quantiles
    def predict(self, Q): return Q

def data(rng, n, K=100):
    x = rng.uniform(0.5, 2, n); y = x * rng.gamma(2.0, 1.0, n)
    base = np.quantile(rng.gamma(2.0, 1.0, 50_000), np.linspace(0.001, 0.999, K + 1))
    Q = np.sort(x[:, None] * base[None, :] * np.exp(0.15 * rng.normal(size=(n, 1))) + 0.05 * rng.normal(size=(n, K + 1)), axis=1)
    return Q, y
def authors(cls, Qc, yc, Qt, alpha=0.1):
    m = cls(Stored()); m.calibrate(Qc, yc, alpha); return m.predict(Qt), m.threshold
def ours(Qc, yc, Qt, alpha=0.1, **kw):
    m = pc.ConformalInterquantileRegressor(alpha=alpha, **kw).calibrate(quantiles=Qc, y=yc); return m.predict_interval(quantiles=Qt)
cov = lambda iv, y: float(((iv[:, 0] <= y) & (y <= iv[:, 1])).mean())
wid = lambda iv: float(np.mean(iv[:, 1] - iv[:, 0]))

rng = np.random.default_rng(0)
print("1. Coverage at a 90% target, 500 repetitions, 2,000 test points each")
for n_cal in (99, 500):
    rows = {k: [] for k in ("authors CIR", "authors CIR+ (CIR_rank)", "pyconformal CIR, same window in calibration and prediction", "pyconformal CIR, window='shortest' (the paper's formula)", "pyconformal CIR+")}
    for _ in range(500):
        Qc, yc = data(rng, n_cal); Qt, yt = data(rng, 2000)
        for name, f in (("authors CIR", lambda: authors(A.CIR, Qc, yc, Qt)[0]), ("authors CIR+ (CIR_rank)", lambda: authors(A.CIR_rank, Qc, yc, Qt)[0]),
                        ("pyconformal CIR, same window in calibration and prediction", lambda: ours(Qc, yc, Qt)), ("pyconformal CIR, window='shortest' (the paper's formula)", lambda: ours(Qc, yc, Qt, window="shortest")),
                        ("pyconformal CIR+", lambda: ours(Qc, yc, Qt, plus=True))):
            try: iv = f(); rows[name].append((cov(iv, yt), wid(iv)))
            except Exception as e: rows[name].append((np.nan, np.nan)); err = f"{type(e).__name__}: {e}"
    print(f"  {n_cal} calibration points")
    for name, v in rows.items():
        v = np.array(v); bad = int(np.isnan(v[:, 0]).sum())
        print(f"    {name:62} coverage {np.nanmean(v[:, 0]):.4f} (s.e. {np.nanstd(v[:, 0]) / np.sqrt(len(v) - bad):.4f})  mean length {np.nanmean(v[:, 1]):.3f}" + (f"  failed {bad} times ({err})" if bad else ""))

print("\n2. Same inputs, 20,000 calibration and 200,000 test points")
Qc, yc = data(rng, 20_000); Qt, yt = data(rng, 200_000)
def best(f, reps):
    b, r = 1e9, None
    for _ in range(reps):
        t = time.perf_counter(); r = f(); b = min(b, time.perf_counter() - t)
    return b, r
ours(Qc[:50], yc[:50], Qt[:50]); ours(Qc[:50], yc[:50], Qt[:50], plus=True); ours(Qc[:50], yc[:50], Qt[:50], window="shortest")
ta, (iva, tha) = best(lambda: authors(A.CIR, Qc, yc, Qt), 2)
to, ivo = best(lambda: ours(Qc, yc, Qt), 5); ts, ivs = best(lambda: ours(Qc, yc, Qt, window="shortest"), 5)
same = np.isclose(iva, ivs).all(axis=1).mean()
print(f"  CIR   authors {ta:7.2f} s  coverage {cov(iva, yt):.4f}  length {wid(iva):.3f}  (number of intervals used: {tha:.0f})")
print(f"        pyconformal, window='shortest' {ts * 1e3:6.0f} ms  coverage {cov(ivs, yt):.4f}  length {wid(ivs):.3f}  identical to the authors' on {same:.1%} of test points  ({ta / ts:.0f}x)")
print(f"        pyconformal, default           {to * 1e3:6.0f} ms  coverage {cov(ivo, yt):.4f}  length {wid(ivo):.3f}  ({ta / to:.0f}x)")
ta, (iva, tha) = best(lambda: authors(A.CIR_rank, Qc, yc, Qt), 1)
to, ivo = best(lambda: ours(Qc, yc, Qt, plus=True), 5)
print(f"  CIR+  authors {ta:7.2f} s  coverage {cov(iva, yt):.4f}  length {wid(iva):.3f}")
print(f"        pyconformal                    {to * 1e3:6.0f} ms  coverage {cov(ivo, yt):.4f}  length {wid(ivo):.3f}  ({ta / to:.0f}x)")
