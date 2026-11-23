"""crepes 0.9.1, core objects on i.i.d. data with a fixed predictor (prediction 0, residual = y).
(a) coverage of every interval and set type; (b) exact rank; (c) a Mondrian category absent from calibration; (d) seed and the global random state."""
import sys, math, warnings, numpy as np
from fractions import Fraction
warnings.simplefilter("ignore")
from crepes import ConformalRegressor, ConformalPredictiveSystem, ConformalClassifier
R = int(sys.argv[1]); ntest = 200
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
print("(a) coverage, mean over", R, "repetitions")
for name, make in (("ConformalRegressor", ConformalRegressor), ("ConformalPredictiveSystem", ConformalPredictiveSystem)):
    for variant in ("standard", "normalized", "Mondrian", "normalized + Mondrian"):
        for n in (19, 49, 199, 1000):
            for conf in (0.9, 0.95, 0.99):
                cov = []; inf = 0
                for r in range(R):
                    g = np.random.default_rng(1000003 * r + n); N = n + ntest
                    sig = np.exp(g.normal(0, 0.5, N)) if "normalized" in variant else None
                    bins = g.integers(0, 3, N) if "Mondrian" in variant else None
                    scale = (sig if sig is not None else 1.0) * (np.array([1.0, 2.0, 4.0])[bins] if bins is not None else 1.0)
                    y = g.normal(size=N) * scale
                    kw_c = dict(sigmas=None if sig is None else sig[:n], bins=None if bins is None else bins[:n])
                    kw_t = dict(sigmas=None if sig is None else sig[n:], bins=None if bins is None else bins[n:])
                    iv = make().fit(y[:n], **kw_c).predict_int(np.zeros(ntest), confidence=conf, **kw_t)
                    cov.append(((iv[:, 0] <= y[n:]) & (y[n:] <= iv[:, 1])).mean()); inf += np.isinf(iv).any(1).mean()
                m, se = np.mean(cov), np.std(cov) / np.sqrt(R)
                print(f"  {name:26s} {variant:22s} n={n:4d} level {conf}: {m:.4f} (se {se:.4f}) {verdict(m, se, conf)}; infinite {inf/R:.2f}", flush=True)
K = 4
for variant in ("standard", "Mondrian", "class-conditional"):
    for smoothing in (True, False):
        for n in (49, 199, 1000):
            cov = []; size = []
            for r in range(R):
                g = np.random.default_rng(7 * r + n); N = n + ntest
                P = g.dirichlet(np.ones(K), N); y = np.array([g.choice(K, p=p) for p in P]); A = 1 - P     # hinge score for every class
                bins = g.integers(0, 3, N)
                a_cal = A[np.arange(n), y[:n]]
                if variant == "standard": cc = ConformalClassifier().fit(a_cal); S = cc.predict_set(A[n:], confidence=0.9, smoothing=smoothing)
                elif variant == "Mondrian": cc = ConformalClassifier().fit(a_cal, bins=bins[:n]); S = cc.predict_set(A[n:], bins=bins[n:], confidence=0.9, smoothing=smoothing)
                else:
                    cc = ConformalClassifier().fit(a_cal, bins=y[:n]); S = np.column_stack([cc.predict_set(A[n:, [c]], bins=np.full(ntest, c), confidence=0.9, smoothing=smoothing)[:, 0] for c in range(K)])
                S = np.asarray(S); cov.append(S[np.arange(ntest), y[n:]].mean()); size.append(S.sum(1).mean())
            m, se = np.mean(cov), np.std(cov) / np.sqrt(R)
            print(f"  ConformalClassifier        {variant:18s} smoothing={str(smoothing):5s} n={n:4d} level 0.9: {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}; mean set size {np.mean(size):.2f}", flush=True)
print("\n(b) rank used against the exact rank ceil((n+1) * level), residuals 1..n")
for name, make in (("ConformalRegressor", ConformalRegressor),):
    for conf in (0.8, 0.9, 0.95, 0.975, 0.99):
        c = [0, 0, 0, 0]; ex = []
        for n in range(1, 2001):
            k = math.ceil((n + 1) * Fraction(repr(conf)))
            if k > n: continue
            hi = make().fit(np.arange(1.0, n + 1)).predict_int(np.zeros(1), confidence=conf)[0, 1]
            i = 0 if hi == k else (1 if hi == k + 1 else (3 if np.isinf(hi) else 2)); c[i] += 1
            if i and len(ex) < 6: ex.append((n, hi, k))
        print(f"  {name} level {conf}: exact {c[0]}, one rank too high {c[1]}, infinite although a finite rank exists {c[3]}, other {c[2]}; examples (n, used, exact) {ex}")
for conf in (0.8, 0.9, 0.95, 0.99):
    c = [0, 0, 0]; ex = []
    for n in range(1, 2001):
        a = Fraction(1) - Fraction(repr(conf)); kl = math.floor((n + 1) * a / 2); kh = math.ceil((n + 1) * (1 - a / 2))
        if kl < 1 or kh > n: continue
        iv = ConformalPredictiveSystem().fit(np.arange(1.0, n + 1)).predict_int(np.zeros(1), confidence=conf)[0]
        ok = iv[0] == kl and iv[1] == kh; wide = iv[0] <= kl and iv[1] >= kh
        c[0 if ok else (1 if wide else 2)] += 1
        if not ok and len(ex) < 6: ex.append((n, tuple(iv), (kl, kh)))
    print(f"  ConformalPredictiveSystem level {conf}: exact {c[0]}, wider than exact {c[1]}, NARROWER than exact {c[2]}; examples (n, used, exact) {ex}")
print("\n(c) a test object whose Mondrian category is absent from the calibration set")
g = np.random.default_rng(0); res = g.normal(size=300); bins = g.integers(0, 3, 300)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    iv = ConformalRegressor().fit(res, bins=bins).predict_int(np.array([10.0, 10.0]), bins=np.array([1, 7]), confidence=0.9)
    pc = ConformalPredictiveSystem().fit(res, bins=bins).predict_int(np.array([10.0, 10.0]), bins=np.array([1, 7]), confidence=0.9)
    print("  ConformalRegressor: prediction 10, category seen:", np.round(iv[0], 3), "| category unseen:", iv[1], "| warnings raised:", len(w))
    print("  ConformalPredictiveSystem: category seen:", np.round(pc[0], 3), "| category unseen:", pc[1])
print("\n(d) does a call with seed= leave NumPy's global random state changed?")
def after(f):
    np.random.seed(12345); np.random.rand(3); f(); return np.random.rand()
base = after(lambda: None)
cps = ConformalPredictiveSystem().fit(res); cr = ConformalRegressor().fit(res); cc = ConformalClassifier().fit(np.abs(res))
for label, f in (("ConformalPredictiveSystem.predict(y=..., seed=0)", lambda: cps.predict(np.zeros(5), y=np.zeros(5), seed=0)),
                 ("ConformalPredictiveSystem.predict(lower_percentiles=5, seed=0)", lambda: cps.predict(np.zeros(5), lower_percentiles=5, seed=0)),
                 ("ConformalPredictiveSystem.fit(seed=0) then predict(y=...)", lambda: ConformalPredictiveSystem().fit(res, seed=0).predict(np.zeros(5), y=np.zeros(5))),
                 ("ConformalRegressor.predict_p(seed=0)", lambda: cr.predict_p(np.zeros(5), np.zeros(5), seed=0)),
                 ("ConformalClassifier.predict_set(seed=0)", lambda: cc.predict_set(np.abs(res[:5]).reshape(-1, 1), seed=0)),
                 ("ConformalPredictiveSystem.predict(y=...) without seed", lambda: cps.predict(np.zeros(5), y=np.zeros(5)))):
    v = after(f); np.random.seed(0); np.random.rand(5); s0 = np.random.rand()
    print(f"  {label:62s}: next global draw {'unchanged' if v == base else ('RESET to the seeded stream' if v == s0 else 'changed (it consumed draws, as expected without a seed)')}")
