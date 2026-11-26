"""Peak extra memory of one call: pyconformal against other packages.

Each case runs in a fresh process. The inputs are built and the compiled
kernels warmed on 3,000 rows first; the process's peak-memory counter is then reset
(Linux, /proc/self/clear_refs, after returning freed memory to the system) and read again after the call, so the figure
is the most memory the call held beyond what was already there.
Usage: python memory.py            (runs every case)
"""
import subprocess, sys, warnings; warnings.filterwarnings("ignore")
import numpy as np

def rss(field):
    for line in open("/proc/self/status"):
        if line.startswith(field): return int(line.split()[1]) / 1024  # MB
def measure(f):
    import gc, time, ctypes; gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
    open("/proc/self/clear_refs", "w").write("5")
    base = rss("VmRSS"); t = time.perf_counter(); f(); dt = time.perf_counter() - t
    return rss("VmHWM") - base, dt

def cvplus(lib, n, nt):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    rng = np.random.default_rng(0)
    def data(n, d=10):
        X = rng.normal(size=(n, d)); return X, X @ np.arange(1, d + 1) + rng.standard_t(4, size=n)
    X, y = data(n); Xt, _ = data(nt)
    if lib == "pyconformal":
        import pyconformal as pc
        pc.CrossConformalRegressor(LinearRegression(), cv=10).fit(*data(3000)).predict_interval(data(3000)[0])
        m = pc.CrossConformalRegressor(LinearRegression(), alpha=0.1, cv=10, random_state=0).fit(X, y)
        return measure(lambda: m.predict_interval(Xt))
    from mapie.regression import CrossConformalRegressor as Mapie
    m = Mapie(LinearRegression(), confidence_level=0.9, method="plus", cv=KFold(10, shuffle=True, random_state=0)).fit_conformalize(X, y)
    return measure(lambda: m.predict_interval(Xt))

def cps(lib, task, n, m):
    rng = np.random.default_rng(0)
    yc, pc_, sc = rng.normal(size=n), np.zeros(n), rng.uniform(0.5, 3, n)
    pt, st = rng.normal(size=m), rng.uniform(0.5, 3, m); yt = pt + st * rng.normal(size=m)
    if lib == "pyconformal":
        from pyconformal import ConformalPredictiveSystem as Ours
        w = Ours().calibrate(y_pred=pc_[:3000], y=yc[:3000], sigma=sc[:3000]); w.predict_interval(y_pred=pt[:3000], alpha=0.1, sigma=st[:3000]); w.predict_cdf(y_pred=pt[:3000], y=yt[:3000], sigma=st[:3000], tau=1.0)
        if task == "cdf": Ours().calibrate(y_pred=pc_, y=yc, sigma=sc).predict_cdf(y_pred=pt[: m // 2], y=yt[: m // 2], sigma=st[: m // 2], tau=1.0)  # the large-input kernel compiles on first use
        if task == "crps":
            Ours().calibrate(y_pred=pc_[:3000], y=yc[:3000]).crps(y_pred=pt[:3000], y=yt[:3000]); o = Ours().calibrate(y_pred=pc_, y=yc)
            return measure(lambda: o.crps(y_pred=pt, y=yt))
        o = Ours().calibrate(y_pred=pc_, y=yc, sigma=sc)
        if task == "interval": return measure(lambda: o.predict_interval(y_pred=pt, alpha=0.1, sigma=st))
        return measure(lambda: o.predict_cdf(y_pred=pt, y=yt, sigma=st, tau=1.0))
    from crepes import ConformalPredictiveSystem as Ref
    if task == "crps":
        r = Ref().fit(yc - pc_); return measure(lambda: r.evaluate(pt, yt, metrics=["CRPS"]))
    r = Ref().fit(yc - pc_, sigmas=sc)
    if task == "interval": return measure(lambda: r.predict(pt, sigmas=st, lower_percentiles=[5], higher_percentiles=[95]))
    return measure(lambda: r.predict(pt, sigmas=st, y=yt, smoothing=False))

def va(lib, n, m):
    rng = np.random.default_rng(0)
    s = rng.uniform(size=n); y = (rng.uniform(size=n) < s ** 1.5).astype(int); t = rng.uniform(size=m)
    P, T = np.column_stack([1 - s, s]), np.column_stack([1 - t, t])
    if lib == "pyconformal":
        from pyconformal import VennAbers
        VennAbers().calibrate(scores=s[:3000], y=y[:3000]).predict_p0_p1(scores=t[:3000])
        return measure(lambda: VennAbers().calibrate(scores=s, y=y).predict_p0_p1(scores=t))
    from venn_abers import VennAbers as PVA
    def f():
        v = PVA(); v.fit(P, y); return v.predict_proba(T)
    return measure(f)

CASES = [
    ("CV+ prediction, 10 folds", "cvplus", "MAPIE", (2000, 20000)), ("CV+ prediction, 10 folds", "cvplus", "MAPIE", (5000, 20000)),
    ("CPS 90% intervals, normalised", "cps", "crepes", ("interval", 10000, 100000)), ("CPS 90% intervals, normalised", "cps", "crepes", ("interval", 100000, 1000000)),
    ("CPS distribution at outcomes", "cps", "crepes", ("cdf", 10000, 100000)), ("CPS distribution at outcomes", "cps", "crepes", ("cdf", 100000, 1000000)),
    ("CPS CRPS", "cps", "crepes", ("crps", 5000, 5000)),
    ("Venn-Abers, fit + predict", "va", "venn-abers", (100000, 1000000)), ("Venn-Abers, fit + predict", "va", "venn-abers", (1000000, 1000000)),
]
if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn, lib, args = sys.argv[1], sys.argv[2], eval(sys.argv[3])
        mb, dt = globals()[fn](lib, *args); print(f"{mb:.1f} {dt:.4f}"); sys.exit()
    print(f"{'case':32} {'sizes':28} {'other':11} {'theirs MB':>10} {'ours MB':>9} {'ratio':>7} {'theirs s':>9} {'ours s':>8}")
    for name, fn, other, args in CASES:
        out = []
        for lib in (other, "pyconformal"):
            r = subprocess.run([sys.executable, __file__, fn, lib, repr(args)], capture_output=True, text=True)
            out.append(tuple(map(float, r.stdout.split())) if r.returncode == 0 and r.stdout.strip() else None)
            if out[-1] is None: err = (r.stderr.strip().splitlines() or ["killed"])[-1][:70]
        a, b = out
        if a and b: print(f"{name:32} {str(args):28} {other:11} {a[0]:10.1f} {b[0]:9.1f} {a[0] / max(b[0], 0.1):6.0f}x {a[1]:9.3f} {b[1]:8.4f}", flush=True)
        else: print(f"{name:32} {str(args):28} {other:11} {'failed' if a is None else f'{a[0]:.1f}':>10} {'failed' if b is None else f'{b[0]:.1f}':>9}   ({err})", flush=True)
