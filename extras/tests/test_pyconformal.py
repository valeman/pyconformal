import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from pyconformal import (
    SplitConformalClassifier,
    SplitConformalRegressor,
    conformal_quantile,
)


# ---------------------------------------------------------------- quantile


def test_quantile_picks_kth_smallest():
    scores = np.arange(1, 20)  # n = 19
    # k = ceil(20 * 0.95) = 19 -> 19th smallest = 19
    assert conformal_quantile(scores, 0.05) == 19.0
    # k = ceil(20 * 0.9) = 18 -> 18
    assert conformal_quantile(scores, 0.1) == 18.0


def test_quantile_is_infinite_when_calibration_set_too_small():
    # n = 5, alpha = 0.1: k = ceil(6 * 0.9) = 6 > 5
    assert math.isinf(conformal_quantile([1, 2, 3, 4, 5], 0.1))


def test_quantile_ignores_input_order():
    rng = np.random.default_rng(0)
    s = rng.normal(size=101)
    assert conformal_quantile(s, 0.2) == conformal_quantile(np.sort(s), 0.2)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_quantile_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], alpha)


def test_quantile_rejects_empty_and_nonfinite():
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, np.nan], 0.1)


# --------------------------------------------------------------- coverage


def _empirical_coverage(make_model, n_trials=200, n_cal=200, alpha=0.1):
    rng = np.random.default_rng(42)
    hits = []
    for _ in range(n_trials):
        hits.append(make_model(rng, n_cal, alpha))
    return float(np.mean(hits))


def test_regressor_coverage():
    def trial(rng, n_cal, alpha):
        n_train, n_test = 200, 500
        n = n_train + n_cal + n_test
        X = rng.normal(size=(n, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_t(df=3, size=n)
        model = SplitConformalRegressor(LinearRegression(), alpha=alpha)
        model.fit(X[:n_train], y[:n_train])
        model.calibrate(X[n_train:n_train + n_cal], y[n_train:n_train + n_cal])
        iv = model.predict_interval(X[n_train + n_cal:])
        yt = y[n_train + n_cal:]
        return np.mean((iv[:, 0] <= yt) & (yt <= iv[:, 1]))

    cov = _empirical_coverage(trial)
    # Theory: 0.9 <= E[coverage] <= 0.9 + 1/201 ~ 0.905
    assert 0.89 <= cov <= 0.915


def test_classifier_coverage():
    def trial(rng, n_cal, alpha):
        n_train, n_test = 300, 500
        n = n_train + n_cal + n_test
        y = rng.integers(0, 3, size=n)
        X = rng.normal(size=(n, 2)) + np.array([[0, 0], [1.5, 0], [0, 1.5]])[y]
        model = SplitConformalClassifier(LogisticRegression(), alpha=alpha)
        model.fit(X[:n_train], y[:n_train])
        model.calibrate(X[n_train:n_train + n_cal], y[n_train:n_train + n_cal])
        sets = model.predict_set(X[n_train + n_cal:])
        yt = y[n_train + n_cal:]
        col = np.searchsorted(model.classes_, yt)
        return np.mean(sets[np.arange(len(yt)), col])

    cov = _empirical_coverage(trial)
    assert 0.89 <= cov <= 0.915


# ------------------------------------------------------------------- API


def test_regressor_interval_shape_and_alpha_override():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2))
    y = X.sum(axis=1) + rng.normal(size=300)
    m = SplitConformalRegressor(LinearRegression()).fit(X[:100], y[:100])
    m.calibrate(X[100:200], y[100:200])
    narrow = m.predict_interval(X[200:], alpha=0.5)
    wide = m.predict_interval(X[200:], alpha=0.05)
    assert narrow.shape == (100, 2)
    assert np.all(wide[:, 1] - wide[:, 0] >= narrow[:, 1] - narrow[:, 0])


def test_trained_model_and_errors():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))
    y = X[:, 0] + rng.normal(size=100)
    est = LinearRegression().fit(X[:50], y[:50])
    m = SplitConformalRegressor(est)
    with pytest.raises(RuntimeError):
        m.predict_interval(X)
    m.calibrate(X[50:], y[50:])
    assert m.predict_interval(X).shape == (100, 2)


def test_classifier_rejects_unknown_calibration_label():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    m = SplitConformalClassifier(LogisticRegression()).fit(X, y)
    with pytest.raises(ValueError):
        m.calibrate(X[:3], np.array([0, 1, 7]))
