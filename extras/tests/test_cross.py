import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

import pyconformal as pc


def make(rng, n, d=5):
    X = rng.normal(size=(n, d))
    return X, X @ np.arange(1, d + 1) + rng.standard_t(4, size=n)


def brute(P, oof_resid, folds, alpha, method="plus"):
    """The definition in Barber et al. (2021), written out."""
    n = len(oof_resid)
    j_lo, j_hi = math.floor(alpha * (n + 1)), math.ceil((1 - alpha) * (n + 1))
    out = np.empty((P.shape[0], 2))
    for i in range(P.shape[0]):
        mu = P[i, folds]
        if method == "minmax":
            q = np.sort(oof_resid)[j_hi - 1] if j_hi <= n else np.inf
            out[i] = P[i].min() - q, P[i].max() + q
            continue
        out[i, 0] = np.sort(mu - oof_resid)[j_lo - 1] if j_lo >= 1 else -np.inf
        out[i, 1] = np.sort(mu + oof_resid)[j_hi - 1] if j_hi <= n else np.inf
    return out


@pytest.mark.parametrize("n,K", [(400, 5), (400, 40), (97, 97), (60, 3), (8, 2)])
@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.5])
def test_matches_definition(n, K, alpha):
    rng = np.random.default_rng(n + K)
    y, oof = rng.normal(size=n), rng.normal(size=n) * 0.5
    oof[: n // 4] = np.round(oof[: n // 4], 1)  # ties
    y[: n // 4] = np.round(y[: n // 4], 1)
    folds = rng.permutation(n) % K
    P = rng.normal(size=(50, K))
    for method in ("plus", "minmax"):
        cp = pc.CrossConformalRegressor(alpha=alpha, method=method).calibrate(y_pred=oof, y=y, folds=folds)
        np.testing.assert_array_equal(cp.predict_interval(y_pred=P), brute(P, np.abs(y - oof), folds, alpha, method))


def test_compiled_and_numpy_paths_agree():
    from pyconformal import cross as C

    if not C.HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(5)
    y, oof, folds = rng.normal(size=3000), rng.normal(size=3000), rng.permutation(3000) % 10
    P = rng.normal(size=(700, 10))
    cp = pc.CrossConformalRegressor().calibrate(y_pred=oof, y=y, folds=folds)
    fast = cp.predict_interval(y_pred=P, alpha=[0.2, 0.1, 0.01])
    C.HAVE_NUMBA = False
    try:
        slow = cp.predict_interval(y_pred=P, alpha=[0.2, 0.1, 0.01])
    finally:
        C.HAVE_NUMBA = True
    np.testing.assert_array_equal(fast, slow)


@pytest.mark.parametrize("cv,method", [(10, "plus"), ("loo", "plus"), (10, "minmax")])
def test_coverage_from_a_model(cv, method):
    rng = np.random.default_rng(7)
    cov, width = [], []
    for rep in range(150):
        X, y = make(rng, 60)
        Xt, yt = make(rng, 200)
        cp = pc.CrossConformalRegressor(LinearRegression(), alpha=0.1, cv=cv, method=method, random_state=rep).fit(X, y)
        iv = cp.predict_interval(Xt)
        cov.append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean()); width.append(np.mean(iv[:, 1] - iv[:, 0]))
    cov = np.array(cov)
    floor = 0.9 if method == "minmax" else 0.8  # the theorems
    assert cov.mean() >= floor
    assert cov.mean() >= 0.9 - 3 * cov.std() / math.sqrt(len(cov))  # and in practice the target
    assert cov.mean() < 0.96


def test_uses_all_data_so_intervals_beat_split_on_small_samples():
    rng = np.random.default_rng(8)
    w_cv, w_split = [], []
    for rep in range(100):
        X, y = make(rng, 40, d=8)
        Xt, _ = make(rng, 100, d=8)
        cv = pc.CrossConformalRegressor(Ridge(1.0), cv=10, random_state=rep).fit(X, y).predict_interval(Xt)
        sp = pc.SplitConformalRegressor(Ridge(1.0)).fit(X[:20], y[:20]).calibrate(X[20:], y[20:]).predict_interval(Xt)
        w_cv.append(np.mean(cv[:, 1] - cv[:, 0])); w_split.append(np.mean(sp[:, 1] - sp[:, 0]))
    assert np.mean(w_cv) < 0.8 * np.mean(w_split)


def test_both_routes_and_errors():
    rng = np.random.default_rng(9)
    X, y = make(rng, 200)
    Xt, _ = make(rng, 30)
    cp = pc.CrossConformalRegressor(LinearRegression(), cv=5, random_state=0).fit(X, y)
    assert len(cp.estimators_) == 5 and cp.predict(Xt).shape == (30,)
    P = np.column_stack([m.predict(Xt) for m in cp.estimators_])
    np.testing.assert_array_equal(cp.predict_interval(Xt), cp.predict_interval(y_pred=P))
    assert cp.predict_interval(Xt, alpha=[0.2, 0.05]).shape == (30, 2, 2)
    tiny = pc.CrossConformalRegressor(alpha=0.01).calibrate(y_pred=np.zeros(10), y=np.arange(10.0), folds=np.arange(10) % 2)
    assert np.all(np.isinf(tiny.predict_interval(y_pred=np.zeros((3, 2)))))  # 10 points cannot give 99%
    free = pc.CrossConformalRegressor()
    with pytest.raises(TypeError):
        free.fit(X, y)
    with pytest.raises(TypeError):
        free.calibrate(X, y)
    with pytest.raises(ValueError):
        free.calibrate(y_pred=y, y=y, folds=np.zeros(200))
    with pytest.raises(ValueError):
        cp.predict_interval(y_pred=P[:, :3])
    with pytest.raises(RuntimeError):
        free.predict_interval(y_pred=P)
    with pytest.raises(ValueError):
        pc.CrossConformalRegressor(method="base")
    with pytest.raises(ValueError):
        pc.CrossConformalRegressor(LinearRegression(), cv=1).fit(X, y)
