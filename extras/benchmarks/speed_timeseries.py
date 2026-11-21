"""Speed of the online, ensemble and cross methods against other libraries
and against the code published with the papers.

Usage: python speed_timeseries.py [folder holding clones of
  aangelopoulos/conformal-time-series and isgibbs/DtACI] [path to Rscript]
Sections whose reference is not installed are skipped and say so.
Times are for the conformal step on the same inputs; "per step" divides by
the number of time steps, because several references are too slow to run at
the same length as pyconformal.
"""
import os, subprocess, sys, tempfile, time, types, warnings, logging, io, contextlib
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
import pyconformal as pc

REF = sys.argv[1] if len(sys.argv) > 1 else ""
RSCRIPT = sys.argv[2] if len(sys.argv) > 2 else ""
rng = np.random.default_rng(0)

def timed(f, reps=1):
    best, out = 1e18, None
    for _ in range(reps):
        t = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t)
    return best, out
def row(method, ref, n_ref, t_ref, n_us, t_us, note=""):
    a, b = t_ref / n_ref * 1e6, t_us / n_us * 1e6
    print(f"{method:22} {ref:42} {a:12.2f} {b:12.3f} {a / b:9.0f}x  {note}")
def skip(method, ref, e):
    print(f"{method:22} {ref:42} not run: {type(e).__name__}: {str(e)[:60]}")
def drift(T):
    return np.abs(rng.normal(size=T)) * np.where((np.arange(T) // 2000) % 2 == 0, 1.0, 3.0)

print(f"{'method':22} {'reference':42} {'ref us/step':>12} {'ours us/step':>12} {'faster':>10}")
big = drift(1_000_000)
for m in (pc.ConformalPID(), pc.AdaptiveConformalInference(gamma=0.005), pc.DtACI(), pc.AgACI(), pc.NonexchangeableConformal(rho=0.99)):
    m.run(scores=big[:2000])  # compile
ours = {}
for name, make in (("pid", lambda: pc.ConformalPID(lr=0.1)), ("pid_i", lambda: pc.ConformalPID(lr=0.1, KI=1.0)),
                   ("aci", lambda: pc.AdaptiveConformalInference(gamma=0.005)), ("dtaci", lambda: pc.DtACI()),
                   ("agaci", lambda: pc.AgACI()), ("nexcp", lambda: pc.NonexchangeableConformal(rho=0.99))):
    ours[name] = timed(lambda: make().run(scores=big), 3)[0]

# ---- conformal PID and ACI: code published with Angelopoulos, Candes and Tibshirani (2023)
try:
    sys.path.insert(0, os.path.join(REF, "conformal-time-series"))
    try:
        np.infty
    except AttributeError:  # removed in NumPy 2; the authors' code still uses it
        np.infty = np.inf
    with contextlib.redirect_stderr(io.StringIO()):
        from core import methods as A
        s = drift(20_000)
        t, r = timed(lambda: A.quantile_integrator_log(s, 0.1, 0.1, 1.0, 0, 1, 100, proportional_lr=True))
        mine = pc.ConformalPID(lr=0.1).run(scores=s).thresholds
        row("Conformal PID (P)", "authors' code (conformal-time-series)", 20_000, t, 1e6, ours["pid"], f"same thresholds: {np.allclose(r['q'], mine)}")
        t, r = timed(lambda: A.quantile_integrator_log(s, 0.1, 0.1, 1.0, 1.0, 1, 100, proportional_lr=True))
        row("Conformal PID (P+I)", "authors' code (conformal-time-series)", 20_000, t, 1e6, ours["pid_i"])
        t, r = timed(lambda: A.aci(s, 0.1, 0.005, 10**9, 0, 1))
        row("ACI", "authors' code (conformal-time-series)", 20_000, t, 1e6, ours["aci"])
except Exception as e:
    skip("Conformal PID / ACI", "authors' code (conformal-time-series)", e)

# ---- DtACI and AgACI: R code published with Gibbs and Candes (2022)
try:
    if not RSCRIPT: raise FileNotFoundError("no Rscript given")
    T = 50_000; s = drift(T)
    d = tempfile.mkdtemp(); np.savetxt(os.path.join(d, "s.csv"), s)
    # The authors' functions take beta_t (the largest level at which step t is
    # covered) as input. It is computed here in one vectorised line per step,
    # the cheapest way in R; pyconformal's time includes the same work.
    code = f'''source("{os.path.join(REF, "DtACI", "DtACI.R")}"); source("{os.path.join(REF, "DtACI", "AgACI.R")}")
s <- scan("{os.path.join(d, "s.csv")}", quiet=TRUE); n <- length(s); g8 <- 0.001*2^(0:7)
g30 <- c(0,5e-6,5e-5,(1:9)*1e-4,(1:9)*1e-3,(1:9)*1e-2)
t0 <- proc.time()[3]; b <- rep(1, n); for (t in 2:n) b[t] <- 1 - sum(s[1:(t-1)] < s[t])/t; cat(proc.time()[3]-t0, "\\n")
t0 <- proc.time()[3]; r <- conformalAdaptStable(b, 0.1, g8, sigma=1/1000, eta=2.72); cat(proc.time()[3]-t0, "\\n")
t0 <- proc.time()[3]; r <- agaci(b, 0.1, g30); cat(proc.time()[3]-t0, "\\n")'''
    out = [float(v) for v in subprocess.run([RSCRIPT, "-e", code], capture_output=True, text=True, timeout=3000).stdout.split()]
    row("DtACI", "authors' R code (isgibbs/DtACI)", T, out[0] + out[1], 1e6, ours["dtaci"], f"R: {out[1] / T * 1e6:.0f} us/step for the update alone, at 50,000 steps")
    row("AgACI", "R code in isgibbs/DtACI (30 experts)", T, out[0] + out[2], 1e6, ours["agaci"], f"R: {out[2] / T * 1e6:.0f} us/step for the update alone, at 50,000 steps")
except Exception as e:
    skip("DtACI / AgACI", "authors' R code (isgibbs/DtACI)", e)

# ---- NexCP: the definition computed directly at every step
T = 4000; s = drift(T)
def direct():
    q = np.empty(T - 50)
    for t in range(50, T):
        past = s[:t]; w = 0.99 ** np.arange(t, 0, -1.0); o = np.argsort(past)
        cum = np.cumsum(np.append(w[o], 1.0)) / (w.sum() + 1.0)
        q[t - 50] = np.append(past[o], np.inf)[np.searchsorted(cum, 0.9 * (1 - 1e-12))]
    return q
t, q = timed(direct)
mine = pc.NonexchangeableConformal(rho=0.99).calibrate(scores=s[:50]).run(scores=s[50:]).thresholds
row("NexCP", "weighted quantile recomputed each step", T - 50, t, 1e6, ours["nexcp"], f"same thresholds: {np.array_equal(q, mine)}")

# ---- EnbPI: MAPIE and PUNCC
from sklearn.linear_model import LinearRegression
def ts(n):
    X = rng.normal(size=(n, 5)); e = np.zeros(n)
    for i in range(1, n): e[i] = 0.5 * e[i - 1] + rng.normal()
    return X, X @ np.arange(1.0, 6.0) + e
X, y = ts(3000); ntr = 1000
t_us, res = timed(lambda: pc.EnbPI(LinearRegression(), alpha=0.1, n_bootstraps=30, random_state=0).fit(X[:ntr], y[:ntr]).run(X[ntr:], y[ntr:]), 3)
try:
    from mapie.regression import TimeSeriesRegressor
    from mapie.subsample import BlockBootstrap
    nm = 300
    def mapie():
        m = TimeSeriesRegressor(LinearRegression(), method="enbpi", cv=BlockBootstrap(n_resamplings=30, n_blocks=10, overlapping=False, random_state=0), agg_function="mean")
        m.fit(X[:ntr], y[:ntr]); hits = 0
        for i in range(ntr, ntr + nm):
            _, pis = m.predict(X[i : i + 1], confidence_level=0.9, ensemble=True)
            hits += pis[0, 0, 0] <= y[i] <= pis[0, 1, 0]
            m.update(X[i : i + 1], y[i : i + 1], ensemble=True)
        return hits / nm
    t, c = timed(mapie)
    row("EnbPI (fit + stream)", "MAPIE", nm, t, 2000, t_us, f"coverage {c:.3f} / ours {res.coverage:.3f}")
except Exception as e:
    skip("EnbPI", "MAPIE", e)
try:
    from deel.puncc.regression import EnbPI as PEnbPI
    from deel.puncc.api.prediction import BasePredictor
    def puncc():  # batch size 10: PUNCC 0.9.3 raises a TypeError with batch size 1
        m = PEnbPI(BasePredictor(LinearRegression()), B=30, random_state=0); m.fit(X[:ntr], y[:ntr])
        out = m.predict(X[ntr:], alpha=0.1, y_true=y[ntr:], s=10); return ((out[1] <= y[ntr:]) & (y[ntr:] <= out[2])).mean()
    t, c = timed(puncc)
    t10, r10 = timed(lambda: pc.EnbPI(LinearRegression(), alpha=0.1, n_bootstraps=30, batch_size=10, random_state=0).fit(X[:ntr], y[:ntr]).run(X[ntr:], y[ntr:]), 3)
    row("EnbPI, batches of 10", "PUNCC", 2000, t, 2000, t10, f"coverage {c:.3f} / ours {r10.coverage:.3f}")
except Exception as e:
    skip("EnbPI", "PUNCC", e)

# ---- SPCI: the quantile forest refitted at every step, same forest settings
try:
    from quantile_forest import RandomForestQuantileRegressor
    Xs, ys = ts(900); n0, w, steps = 600, 10, 100
    def ours_spci():
        return pc.SPCI(LinearRegression(), n_bootstraps=20, past_window=w, refit_every=1, random_state=0).fit(Xs[:n0], ys[:n0]).run(Xs[n0 : n0 + steps], ys[n0 : n0 + steps])
    ours_spci(); t_us2, r2 = timed(ours_spci)
    base = pc.SPCI(LinearRegression(), n_bootstraps=20, past_window=w, random_state=0).fit(Xs[:n0], ys[:n0])
    def with_zillow():  # the paper's loop with the quantile-forest package in place of ours
        r = base.residuals_.copy(); betas = np.linspace(0, 0.1, 6)[:-1]; lv = list(betas) + list(0.9 + betas)
        for i in range(steps):
            win = np.lib.stride_tricks.sliding_window_view(r, w)[:-1]
            f = RandomForestQuantileRegressor(20, min_samples_leaf=5, max_features=1 / 3, random_state=0, max_samples_leaf=None).fit(win, r[w:])
            f.predict(r[-w:][None, :], quantiles=lv)
            r = np.append(r[1:], 0.0)
    t, _ = timed(with_zillow)
    row("SPCI (refit each step)", "same loop with quantile-forest (Zillow)", steps, t, steps, t_us2, "bootstrap fit included in ours only")
except Exception as e:
    skip("SPCI", "same loop with quantile-forest", e)

# ---- cross Venn-Abers
try:
    from venn_abers import VennAbersCalibrator
    from sklearn.naive_bayes import GaussianNB
    n = 100_000; Xc = rng.normal(size=(n, 8)); yc = (rng.uniform(size=n) < 1 / (1 + np.exp(-Xc[:, 0] + Xc[:, 1]))).astype(int); Xt = rng.normal(size=(n, 8))
    pc.CrossVennAbers(GaussianNB(), cv=5, random_state=0).fit(Xc[:500], yc[:500])
    t_us3, p = timed(lambda: pc.CrossVennAbers(GaussianNB(), cv=5, random_state=0).fit(Xc, yc).predict_proba(Xt), 3)
    t, q = timed(lambda: VennAbersCalibrator(estimator=GaussianNB(), inductive=False, n_splits=5, random_state=0).fit(Xc, yc).predict_proba(Xt))
    row("Cross Venn-Abers", "venn-abers (100,000 rows, 5 folds)", 1, t, 1, t_us3, f"mean abs difference {np.abs(p[:, 1] - q[:, 1]).mean():.4f} (folds differ)")
except Exception as e:
    skip("Cross Venn-Abers", "venn-abers", e)

# ---- normalized and Mondrian split regression
try:
    from crepes import ConformalRegressor
    n, m = 100_000, 1_000_000
    yc, pcal, sc, gc = rng.normal(size=n), rng.normal(size=n), rng.uniform(0.5, 2, n), rng.integers(0, 20, n)
    pt, st, gt = rng.normal(size=m), rng.uniform(0.5, 2, m), rng.integers(0, 20, m)
    for label, kw_c, kw_p, ck_c, ck_p in (("Normalized split", dict(sigma=sc), dict(sigma=st), dict(sigmas=sc), dict(sigmas=st)),
                                           ("Mondrian split", dict(groups=gc), dict(groups=gt), dict(bins=gc), dict(bins=gt))):
        t_us4, a = timed(lambda: pc.SplitConformalRegressor().calibrate(y_pred=pcal, y=yc, **kw_c).predict_interval(y_pred=pt, **kw_p), 3)
        t, b = timed(lambda: ConformalRegressor().fit(yc - pcal, **ck_c).predict_int(pt, confidence=0.9, **ck_p), 3)
        row(label, "crepes (100,000 / 1,000,000 rows)", 1, t, 1, t_us4, f"rows with the same interval: {np.isclose(a, b).all(axis=1).mean():.1%}")
except Exception as e:
    skip("Normalized / Mondrian", "crepes", e)
print("\nus/step = microseconds per time step (for the last three rows: microseconds for the whole task).")
