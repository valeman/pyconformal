import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

import pyconformal as pc
from pyconformal import forest as _forest

needs_numba = pytest.mark.skipif(not _forest.HAVE_NUMBA, reason="SPCI uses QuantileForest, which needs numba")


def ar_series(rng, n, phi=0.0, het=False):
    """y = linear signal + noise; noise is autocorrelated (phi) and can have
    slowly changing scale (het)."""
    X = rng.normal(size=(n, 3))
    e = np.zeros(n)
    scale = 1.0 + (2.0 * (np.sin(np.arange(n) / 60.0) > 0) if het else np.zeros(n))
    for t in range(1, n):
        e[t] = phi * e[t - 1] + scale[t] * rng.normal()
    return X, X @ np.array([1.0, -1.0, 0.5]) + e


def test_enbpi_residuals_are_out_of_bag():
    rng = np.random.default_rng(0)
    X, y = ar_series(rng, 150)
    m = pc.EnbPI(LinearRegression(), n_bootstraps=20, random_state=1).fit(X, y)
    # rebuild by hand from the same bootstrap draws
    g = np.random.default_rng(1)
    loo_sum, cnt = np.zeros(150), np.zeros(150)
    for b in range(20):
        idx = g.integers(0, 150, size=150)
        out = np.ones(150, dtype=bool)
        out[idx] = False
        np.testing.assert_allclose(m.estimators_[b].coef_, LinearRegression().fit(X[idx], y[idx]).coef_)
        loo_sum[out] += m.estimators_[b].predict(X[out])
        cnt[out] += 1
    assert cnt.min() > 0 and m.n_starved_ == 0
    np.testing.assert_allclose(m.residuals_, np.abs(y - loo_sum / cnt), atol=1e-12)
    P = np.column_stack([e.predict(X[:5]) for e in m.estimators_])
    np.testing.assert_allclose(m.predict(X[:5]), P.mean(axis=1))
    np.testing.assert_array_equal(m.predict_interval(X[:5]), m.predict_interval(y_pred=P))
    q = pc.conformal_quantile(m.residuals_, 0.1)
    np.testing.assert_allclose(m.predict_interval(X[:5]), np.column_stack([P.mean(1) - q, P.mean(1) + q]))


@pytest.mark.parametrize("batch", [1, 7])
def test_enbpi_run_equals_stepwise(batch):
    rng = np.random.default_rng(1)
    X, y = ar_series(rng, 420, het=True)
    a = pc.EnbPI(Ridge(1.0), n_bootstraps=15, batch_size=batch, random_state=0).fit(X[:200], y[:200])
    res = a.run(X[200:], y[200:])
    b = pc.EnbPI(Ridge(1.0), n_bootstraps=15, batch_size=batch, random_state=0).fit(X[:200], y[:200])
    for s in range(200, 420, batch):
        e = min(s + batch, 420)
        np.testing.assert_allclose(b.predict_interval(X[s:e]), res.intervals[s - 200 : e - 200], atol=1e-12)
        b.update(X[s:e], y[s:e])
    np.testing.assert_allclose(a.residuals_, b.residuals_, atol=1e-12)
    assert len(a.residuals_) == 200 and res.predictions.shape == (220,)


def test_enbpi_coverage_and_adaptation():
    rng = np.random.default_rng(2)
    steady, cov, cov_fixed = [], [], []
    for rep in range(30):
        X, y = ar_series(rng, 1300)
        steady.append(pc.EnbPI(LinearRegression(), n_bootstraps=20, random_state=rep).fit(X[:300], y[:300]).run(X[300:], y[300:]).coverage)
        X, y = ar_series(rng, 1300, het=True)  # noise level switches between 1 and 3
        m = pc.EnbPI(LinearRegression(), n_bootstraps=20, random_state=rep).fit(X[:300], y[:300])
        fixed = m.predict_interval(X[300:])  # never updated
        res = m.run(X[300:], y[300:])
        cov.append(res.coverage)
        cov_fixed.append(((fixed[:, 0] <= y[300:]) & (y[300:] <= fixed[:, 1])).mean())
    assert abs(np.mean(steady) - 0.9) < 0.015  # stable errors: on target
    assert np.std(cov) < np.std(cov_fixed)  # sliding residuals keep coverage steadier under change
    assert 0.84 < np.mean(cov) < 0.92  # but EnbPI's guarantee is only approximate when the noise level moves


@needs_numba
def test_spci_beats_enbpi_when_errors_are_autocorrelated():
    rng = np.random.default_rng(3)
    cov_s, cov_e, w_s, w_e = [], [], [], []
    for rep in range(6):
        X, y = ar_series(rng, 900, phi=0.9)
        e = pc.EnbPI(LinearRegression(), n_bootstraps=15, random_state=rep).fit(X[:600], y[:600]).run(X[600:], y[600:])
        s = pc.SPCI(LinearRegression(), n_bootstraps=15, past_window=10, refit_every=10, random_state=rep).fit(X[:600], y[:600]).run(X[600:], y[600:])
        cov_s.append(s.coverage); cov_e.append(e.coverage)
        w_s.append(np.mean(s.intervals[:, 1] - s.intervals[:, 0])); w_e.append(np.mean(e.intervals[:, 1] - e.intervals[:, 0]))
    assert np.mean(cov_s) > 0.85 and np.mean(cov_e) > 0.85
    assert np.mean(w_s) < 0.7 * np.mean(w_e)  # the next error is predictable from the last ones


@needs_numba
def test_spci_stepwise_and_errors():
    rng = np.random.default_rng(4)
    X, y = ar_series(rng, 330, phi=0.7)
    a = pc.SPCI(LinearRegression(), n_bootstraps=10, past_window=8, random_state=0).fit(X[:300], y[:300])
    res = a.run(X[300:], y[300:])
    b = pc.SPCI(LinearRegression(), n_bootstraps=10, past_window=8, random_state=0).fit(X[:300], y[:300])
    for t in range(300, 330):
        np.testing.assert_allclose(b.predict_interval(X[t : t + 1])[0], res.intervals[t - 300], atol=1e-12)
        b.update(X[t : t + 1], y[t : t + 1])
    assert np.all(res.intervals[:, 0] < res.intervals[:, 1]) and a.past_window_ == 8
    assert b.predict_interval(X[:2], alpha=[0.2, 0.1]).shape == (2, 2, 2)
    with pytest.raises(ValueError):
        pc.SPCI(LinearRegression(), past_window=500).fit(X[:300], y[:300])
    with pytest.raises(TypeError):
        pc.EnbPI().fit(X, y)
    with pytest.raises(RuntimeError):
        pc.EnbPI().predict_interval(y_pred=np.zeros((2, 3)))
    free = pc.EnbPI().calibrate(y_pred=rng.normal(size=(50, 4)), y=rng.normal(size=50), in_bag=rng.random((50, 4)) < 0.6)
    assert free.predict_interval(y_pred=rng.normal(size=(3, 4))).shape == (3, 2)
    with pytest.raises(ValueError):
        free.predict_interval(y_pred=rng.normal(size=(3, 5)))
