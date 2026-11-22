"""QuantileBooster: quantile predictions from CatBoost, XGBoost and LightGBM
for CQR, CTI and CIR. Each library is tested when it is installed."""

import copy

import numpy as np
import pyconformal as pc
import pytest

PARAMS = {
    "catboost": dict(iterations=60, random_seed=0, thread_count=1),
    "xgboost": dict(n_estimators=60, max_depth=4, random_state=0, n_jobs=1),
    "lightgbm": dict(n_estimators=60, num_leaves=15, random_state=0, n_jobs=1),
}


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = X[:, 0] + np.exp(0.5 * X[:, 1]) * rng.gamma(2.0, 1.0, n)
    return X, y


@pytest.fixture(params=sorted(PARAMS))
def lib(request):
    pytest.importorskip(request.param)
    return request.param


def test_shapes_sorted_rows_and_trained_levels(lib):
    X, y = _data()
    qb = pc.QuantileBooster(lib, levels=9, **PARAMS[lib]).fit(X[:800], y[:800])
    np.testing.assert_allclose(qb.levels_, np.arange(1, 10) / 10)
    Q = qb.predict(X[800:], quantiles=np.linspace(0, 1, 51))
    assert Q.shape == (700, 51) and np.all(np.isfinite(Q))
    assert np.all(np.diff(Q, axis=1) >= 0)  # never crossing
    assert qb.predict(X[800:]).shape == (700,)  # the median
    assert qb.predict(X[800:], quantiles=0.3).shape == (700,)
    # at the trained levels, exactly the (sorted) library output
    np.testing.assert_allclose(qb.predict(X[800:], quantiles=qb.levels_), qb._trained(X[800:]))
    # between two trained levels: linear interpolation
    lo, hi = qb.predict(X[800:], quantiles=[0.3, 0.4]).T
    np.testing.assert_allclose(qb.predict(X[800:], quantiles=0.325), lo + 0.25 * (hi - lo))
    # the order of the request is kept
    np.testing.assert_array_equal(qb.predict(X[800:], quantiles=[0.9, 0.1])[:, ::-1], qb.predict(X[800:], quantiles=[0.1, 0.9]))


def test_levels_zero_and_one_contain_every_training_outcome(lib):
    X, y = _data()
    qb = pc.QuantileBooster(lib, levels=9, **PARAMS[lib]).fit(X[:800], y[:800])
    Q = qb.predict(X[:800], quantiles=[0.0, 1.0])
    assert np.all(Q[:, 0] <= y[:800] + 1e-9) and np.all(y[:800] <= Q[:, 1] + 1e-9)


def test_quantiles_are_roughly_calibrated(lib):
    X, y = _data(4000)
    qb = pc.QuantileBooster(lib, levels=19, **PARAMS[lib]).fit(X[:2500], y[:2500])
    for q in (0.2, 0.5, 0.8):
        assert abs((y[2500:] <= qb.predict(X[2500:], quantiles=q)).mean() - q) < 0.08


@pytest.mark.parametrize("make", [
    lambda qb: pc.ConformalizedQuantileRegressor(qb),
    lambda qb: pc.ConformalInterquantileRegressor(qb),
    lambda qb: pc.ConformalInterquantileRegressor(qb, plus=True),
])
def test_conformal_methods_cover_with_a_booster(lib, make):
    X, y = _data(3000)
    qb = pc.QuantileBooster(lib, levels=19, **PARAMS[lib])
    m = make(qb).fit(X[:1000], y[:1000]).calibrate(X[1000:2000], y[1000:2000])
    iv = m.predict_interval(X[2000:])
    cov = ((iv[:, 0] <= y[2000:]) & (y[2000:] <= iv[:, 1])).mean()
    assert 0.86 < cov < 0.96 and np.all(np.isfinite(iv))


def test_cti_with_a_booster(lib):
    X, y = _data(3000)
    qb = pc.QuantileBooster(lib, levels=19, **PARAMS[lib])
    cti = pc.ConformalThresholdedIntervals(qb, quantile_levels=40).fit(X[:1000], y[:1000]).calibrate(X[1000:2000], y[1000:2000])
    iv = cti.predict_interval(X[2000:])  # hull of the set
    cov = ((iv[:, 0] <= y[2000:]) & (y[2000:] <= iv[:, 1])).mean()
    assert cov > 0.86


def test_sample_weight_copy_and_params(lib):
    X, y = _data(600)
    qb = pc.QuantileBooster(lib, levels=[0.1, 0.5, 0.9], **PARAMS[lib])
    qb.fit(X, y, sample_weight=np.ones(600))
    twin = copy.deepcopy(qb)
    np.testing.assert_array_equal(twin.predict(X[:20], quantiles=[0.1, 0.9]), qb.predict(X[:20], quantiles=[0.1, 0.9]))
    assert qb.get_params()["library"] == lib
    assert qb.set_params(levels=5).levels_.size == 5 and qb.models_ is None


def test_bad_arguments():
    with pytest.raises(ValueError):
        pc.QuantileBooster("sklearn")
    with pytest.raises(ValueError):
        pc.QuantileBooster("catboost", levels=1)
    with pytest.raises(ValueError):
        pc.QuantileBooster("catboost", levels=[0.0, 0.5])
    with pytest.raises(RuntimeError):
        pc.QuantileBooster("catboost").predict(np.zeros((1, 2)), quantiles=[0.5])
