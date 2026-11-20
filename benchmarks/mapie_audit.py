"""Coverage audit of MAPIE on i.i.d. data, where the guarantee must hold.
For each configuration: R independent repetitions, coverage measured on fresh test points.
A configuration FAILS if mean coverage is below the guaranteed level by more than 3 standard errors."""
import warnings, sys, time, logging; warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
import numpy as np
from importlib.metadata import version
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import (SplitConformalRegressor, ConformalizedQuantileRegressor,
                              CrossConformalRegressor, JackknifeAfterBootstrapRegressor)
from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
print("mapie", version("mapie"))
N_TEST = 500
def reg_data(rng, n):
    X = rng.normal(size=(n, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + (1 + np.abs(X[:, 0])) * rng.standard_t(4, size=n)
    return X, y
def clf_data(rng, n, K=4):
    y = rng.integers(0, K, size=n); X = rng.normal(size=(n, 3)) + 1.2 * np.eye(K)[y][:, :3]
    return X, y
def report(name, covs, target, note=""):
    covs = np.asarray(covs); m = covs.mean(); se = covs.std(ddof=1) / np.sqrt(len(covs))
    flag = "FAIL" if m < target - 3 * se else "ok"
    print(f"{flag:4} {name:62} target>={target:.3f}  coverage {m:.4f} (se {se:.4f}) {note}", flush=True)
def cov_int(iv, y):  # iv: (n, 2, 1)
    return np.mean((iv[:, 0, 0] <= y) & (y <= iv[:, 1, 0]))

R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for level in (0.9, 0.8):
    for n_cal in (20, 50, 200):
        for score in ("absolute", "gamma", "residual_normalized"):
            for miw in ((False, True) if score == "gamma" else (False,)):
                covs = []
                for r in range(R):
                    rng = np.random.default_rng(1000 * r + n_cal)
                    Xtr, ytr = reg_data(rng, 200); Xc, yc = reg_data(rng, n_cal); Xt, yt = reg_data(rng, N_TEST)
                    if score == "gamma": ytr, yc, yt = ytr + 1000, yc + 1000, yt + 1000
                    try:
                        m = SplitConformalRegressor(LinearRegression().fit(Xtr, ytr), confidence_level=level, conformity_score=score, prefit=True)
                        m.conformalize(Xc, yc)
                        _, iv = m.predict_interval(Xt, minimize_interval_width=miw)
                        covs.append(cov_int(iv, yt))
                    except Exception as e:
                        covs = f"{type(e).__name__}: {str(e)[:70]}"; break
                nm = f"Split regression, score={score}, n_cal={n_cal}, minimize_width={miw}"
                if isinstance(covs, str): print(f"ERR  {nm}: {covs}")
                else: report(nm, covs, level)
    # CQR
    for n_cal in (50, 200):
        for sym in (False, True):
            for miw in (False, True):
                covs = []
                for r in range(max(R // 3, 60)):
                    rng = np.random.default_rng(7000 * r + n_cal)
                    Xtr, ytr = reg_data(rng, 300); Xc, yc = reg_data(rng, n_cal); Xt, yt = reg_data(rng, N_TEST)
                    try:
                        m = ConformalizedQuantileRegressor(QuantileRegressor(alpha=0.0), confidence_level=level)
                        m.fit(Xtr, ytr).conformalize(Xc, yc)
                        _, iv = m.predict_interval(Xt, symmetric_correction=sym, minimize_interval_width=miw)
                        covs.append(cov_int(iv, yt))
                    except Exception as e:
                        covs = f"{type(e).__name__}: {str(e)[:70]}"; break
                nm = f"CQR, n_cal={n_cal}, symmetric={sym}, minimize_width={miw}"
                if isinstance(covs, str): print(f"ERR  {nm}: {covs}")
                else: report(nm, covs, level)
    # Cross conformal: plus/minmax guarantee 1-2a (plus), 1-a (minmax); base: none
    for method, target in (("plus", 1 - 2 * (1 - level)), ("minmax", level), ("base", level)):
        for cv in (5, -1):
            covs = []
            for r in range(max(R // 3, 60)):
                rng = np.random.default_rng(9000 * r + abs(cv))
                X, y = reg_data(rng, 60); Xt, yt = reg_data(rng, N_TEST)
                m = CrossConformalRegressor(LinearRegression(), confidence_level=level, method=method, cv=cv)
                m.fit_conformalize(X, y); _, iv = m.predict_interval(Xt); covs.append(cov_int(iv, yt))
            report(f"Cross regression, method={method}, cv={'LOO' if cv == -1 else cv}, n=60", covs, target, "(base has no guarantee)" if method == "base" else "")
    for method, target in (("plus", 1 - 2 * (1 - level)), ("minmax", level)):
        covs = []
        for r in range(max(R // 3, 60)):
            rng = np.random.default_rng(11000 * r)
            X, y = reg_data(rng, 60); Xt, yt = reg_data(rng, N_TEST)
            m = JackknifeAfterBootstrapRegressor(LinearRegression(), confidence_level=level, method=method, resampling=30, random_state=r)
            m.fit_conformalize(X, y); _, iv = m.predict_interval(Xt); covs.append(cov_int(iv, yt))
        report(f"Jackknife+-after-bootstrap, method={method}, n=60", covs, target)
    # Classification
    for score in ("lac", "aps", "raps", "top_k"):
        for n_cal in (20, 50, 200):
            covs = []; sizes = []
            for r in range(R):
                rng = np.random.default_rng(13000 * r + n_cal)
                Xtr, ytr = clf_data(rng, 300); Xc, yc = clf_data(rng, n_cal); Xt, yt = clf_data(rng, N_TEST)
                try:
                    m = SplitConformalClassifier(LogisticRegression().fit(Xtr, ytr), confidence_level=level, conformity_score=score, prefit=True, random_state=r)
                    m.conformalize(Xc, yc); _, s = m.predict_set(Xt)
                    covs.append(np.mean(s[np.arange(N_TEST), yt, 0])); sizes.append(s[:, :, 0].sum(1).mean())
                except Exception as e:
                    covs = f"{type(e).__name__}: {str(e)[:70]}"; break
            nm = f"Split classification, score={score}, n_cal={n_cal}"
            if isinstance(covs, str): print(f"ERR  {nm}: {covs}")
            else: report(nm, covs, level, f"mean set size {np.mean(sizes):.2f}")
    for score in ("lac", "aps"):
        covs = []
        for r in range(max(R // 3, 60)):
            rng = np.random.default_rng(17000 * r)
            X, y = clf_data(rng, 150); Xt, yt = clf_data(rng, N_TEST)
            m = CrossConformalClassifier(LogisticRegression(), confidence_level=level, conformity_score=score, cv=5, random_state=r)
            m.fit_conformalize(X, y); _, s = m.predict_set(Xt); covs.append(np.mean(s[np.arange(N_TEST), yt, 0]))
        report(f"Cross classification, score={score}, cv=5, n=150", covs, 1 - 2 * (1 - level) if score == "aps" else level, "(guarantee depends on method)")
