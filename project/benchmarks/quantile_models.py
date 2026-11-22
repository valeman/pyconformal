"""Quantile models compared: pyconformal's QuantileForest and QuantileBooster
(CatBoost, XGBoost, LightGBM) against Zillow's quantile-forest.

For each dataset and each of several random splits (train / calibration /
test), every model is trained on the same rows and scored on the same rows:

* pinball loss of its raw quantiles, averaged over levels 0.05 ... 0.95;
* coverage of its raw 5%-95% interval (no conformal step);
* coverage and mean width after CQR, CIR and CIR+ at a 90% target;
* seconds to fit, and to predict 101 quantile levels for the test rows.

    pip install quantile-forest catboost xgboost lightgbm scikit-learn
    python quantile_models.py [n_splits]
"""
import sys
import time
import warnings

import numpy as np
import pyconformal as pc

warnings.filterwarnings("ignore")
LEVELS = np.linspace(0.0, 1.0, 101)
PIN = np.arange(1, 20) / 20.0


def synthetic(kind, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    if kind == "skewed":  # heteroscedastic, right-skewed noise
        y = np.sin(X[:, 0]) + X[:, 1] + np.exp(0.5 * X[:, 2]) * rng.gamma(2.0, 1.0, n)
    else:  # two modes whose separation depends on x
        y = X[:, 0] + np.where(rng.random(n) < 0.5, -1.0, 1.0) * (2.0 + np.abs(X[:, 1])) + 0.3 * rng.normal(size=n)
    return X, y


def datasets():
    from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes

    out = {"synthetic, skewed (20,000 rows)": synthetic("skewed", 20000, 1),
           "synthetic, two modes (20,000 rows)": synthetic("bimodal", 20000, 2)}
    d = fetch_california_housing()
    out["California housing (20,640 rows)"] = (d.data, d.target)
    d = fetch_openml("kin8nm", version=1, as_frame=False)
    out["kin8nm (8,192 rows)"] = (np.asarray(d.data, float), np.asarray(d.target, float))
    d = load_diabetes()
    out["diabetes (442 rows)"] = (d.data, d.target)
    return out


class Zillow:
    """quantile-forest behind the predict(X, quantiles=...) interface."""

    def __init__(self, **kw):
        from quantile_forest import RandomForestQuantileRegressor

        self.m = RandomForestQuantileRegressor(**kw)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X, quantiles=None):
        return self.m.predict(X, quantiles=list(quantiles))


def models(seed, small):
    r = 150 if small else 300
    return {
        "quantile-forest (Zillow)": lambda: Zillow(n_estimators=100, min_samples_leaf=5, max_features=1 / 3, random_state=seed, n_jobs=-1),
        "pyconformal QuantileForest": lambda: pc.QuantileForest(100, min_samples_leaf=5, max_features=1 / 3, random_state=seed),
        "QuantileBooster, CatBoost": lambda: pc.QuantileBooster("catboost", levels=19, iterations=r, learning_rate=0.1, depth=6, random_seed=seed, thread_count=-1),
        "QuantileBooster, XGBoost": lambda: pc.QuantileBooster("xgboost", levels=19, n_estimators=r, learning_rate=0.1, max_depth=6, random_state=seed),
        "QuantileBooster, LightGBM": lambda: pc.QuantileBooster("lightgbm", levels=19, n_estimators=r, learning_rate=0.1, num_leaves=31, min_child_samples=5 if small else 20, random_state=seed),
    }


def pinball(Q, y):
    d = y[:, None] - Q
    return float(np.mean(np.maximum(PIN * d, (PIN - 1.0) * d)))


def main(n_splits=5):
    rows = []
    for dname, (X, y) in datasets().items():
        n = len(y)
        small = n < 2000
        res = {}
        for seed in range(n_splits):
            perm = np.random.default_rng(seed).permutation(n)
            a, b = int(0.5 * n), int(0.75 * n)
            tr, ca, te = perm[:a], perm[a:b], perm[b:]
            for mname, make in models(seed, small).items():
                m = make()
                t = time.perf_counter(); m.fit(X[tr], y[tr]); t_fit = time.perf_counter() - t
                t = time.perf_counter(); Qte = np.asarray(m.predict(X[te], quantiles=LEVELS)); t_pred = time.perf_counter() - t
                Qca = np.asarray(m.predict(X[ca], quantiles=LEVELS))
                idx = np.round(PIN * 100).astype(int)
                raw = ((Qte[:, 5] <= y[te]) & (y[te] <= Qte[:, 95])).mean()
                out = [pinball(Qte[:, idx], y[te]), raw]
                for meth in (pc.ConformalizedQuantileRegressor(), pc.ConformalInterquantileRegressor(),
                             pc.ConformalInterquantileRegressor(plus=True)):
                    if isinstance(meth, pc.ConformalizedQuantileRegressor):
                        iv = meth.calibrate(quantiles=Qca[:, [5, 95]], y=y[ca]).predict_interval(quantiles=Qte[:, [5, 95]])
                    else:
                        iv = meth.calibrate(quantiles=Qca, y=y[ca]).predict_interval(quantiles=Qte)
                    out += [((iv[:, 0] <= y[te]) & (y[te] <= iv[:, 1])).mean(), np.mean(iv[:, 1] - iv[:, 0])]
                res.setdefault(mname, []).append(out + [t_fit, t_pred])
            print(dname, "split", seed, "done", flush=True)
        print(f"\n== {dname}: mean over {n_splits} splits; train {a}, calibration {b - a}, test {n - b}")
        print(f"{'model':28s} pinball  raw90  CQR cov/width    CIR cov/width    CIR+ cov/width   fit s  predict s")
        for mname, v in res.items():
            v = np.mean(v, axis=0)
            print(f"{mname:28s} {v[0]:.4f}  {v[1]:.3f}  {v[2]:.3f} / {v[3]:<8.4g} {v[4]:.3f} / {v[5]:<8.4g} {v[6]:.3f} / {v[7]:<8.4g} {v[8]:6.2f}  {v[9]:.3f}")
            rows.append((dname, mname, *v))
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
