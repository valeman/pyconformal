"""Every method must give the same answer from model outputs as from a model."""
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import pyconformal as pc


@pytest.fixture(scope="module")
def reg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(900, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=900)
    model = LinearRegression().fit(X[:300], y[:300])
    return model, X, y


@pytest.fixture(scope="module")
def clf():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1200, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=1200) > 0).astype(int) + (X[:, 1] > 1)
    model = LogisticRegression(max_iter=500).fit(X[:400], y[:400])
    return model, X, y


def test_split_regression_both_routes(reg):
    model, X, y = reg
    a = pc.SplitConformalRegressor(model, alpha=0.1).calibrate(X[300:600], y[300:600])
    b = pc.SplitConformalRegressor(alpha=0.1).calibrate(y_pred=model.predict(X[300:600]), y=y[300:600])
    np.testing.assert_array_equal(a.predict_interval(X[600:]), b.predict_interval(y_pred=model.predict(X[600:])))
    multi = a.predict_interval(X[600:], alpha=[0.2, 0.1, 0.05])
    assert multi.shape == (300, 2, 3)
    np.testing.assert_array_equal(multi[:, :, 1], a.predict_interval(X[600:]))
    assert np.all(multi[:, 0, 2] <= multi[:, 0, 0]) and np.all(multi[:, 1, 2] >= multi[:, 1, 0])


def test_fit_trains_the_model():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2))
    y = X[:, 0] + rng.normal(size=200)
    cp = pc.SplitConformalRegressor(LinearRegression()).fit(X[:100], y[:100]).calibrate(X[100:], y[100:])
    assert cp.predict_interval(X).shape == (200, 2)
    assert cp.predict(X).shape == (200,)


def test_clear_errors_for_wrong_inputs(reg):
    model, X, y = reg
    free = pc.SplitConformalRegressor()
    with pytest.raises(TypeError, match="y_pred="):
        free.calibrate(X[:10], y[:10])  # X without a model
    with pytest.raises(TypeError, match="no estimator"):
        free.fit(X, y)
    with pytest.raises(TypeError, match="not both"):
        pc.SplitConformalRegressor(model).calibrate(X[:10], y[:10], y_pred=y[:10])
    with pytest.raises(TypeError):
        free.calibrate(y_pred=y[:10])  # no outcomes
    with pytest.raises(RuntimeError):
        free.predict_interval(y_pred=y[:10])


def test_classifier_both_routes(clf):
    model, X, y = clf
    Pc, Pt = model.predict_proba(X[400:800]), model.predict_proba(X[800:])
    for score in ("lac", "aps", "raps", "saps"):
        a = pc.SplitConformalClassifier(model, score=score, random_state=3).calibrate(X[400:800], y[400:800])
        b = pc.SplitConformalClassifier(score=score, random_state=3).calibrate(proba=Pc, y=y[400:800])
        np.testing.assert_array_equal(a.predict_set(X[800:]), b.predict_set(proba=Pt))
    np.testing.assert_array_equal(b.classes_, [0, 1, 2])
    nested = b.predict_set(proba=Pt, alpha=[0.3, 0.1, 0.02])
    assert nested.shape == (400, 3, 3)
    assert np.all(nested[:, :, 0] <= nested[:, :, 1]) and np.all(nested[:, :, 1] <= nested[:, :, 2])
    named = pc.SplitConformalClassifier(classes=["a", "b", "c"]).calibrate(proba=Pc, y=np.array(["a", "b", "c"])[y[400:800]])
    assert set(named.predict(proba=Pt)) <= {"a", "b", "c"}


def test_predictive_system_both_routes(reg):
    model, X, y = reg
    pcal, pte = model.predict(X[300:600]), model.predict(X[600:])
    a = pc.ConformalPredictiveSystem(model).calibrate(X[300:600], y[300:600])
    b = pc.ConformalPredictiveSystem().calibrate(y_pred=pcal, y=y[300:600])
    np.testing.assert_array_equal(a.predict_interval(X[600:]), b.predict_interval(y_pred=pte))
    np.testing.assert_array_equal(a.predict_quantile(X[600:], [0.1, 0.9]), b.predict_quantile(y_pred=pte, q=[0.1, 0.9]))
    np.testing.assert_array_equal(a.predict_cdf(X[600:], y[600:]), b.predict_cdf(y_pred=pte, y=y[600:]))
    np.testing.assert_array_equal(a.crps(X[600:], y[600:]), b.crps(y_pred=pte, y=y[600:]))
    assert b.predict_interval(y_pred=pte, alpha=[0.2, 0.1]).shape == (300, 2, 2)


def test_venn_abers_both_routes(clf):
    model, X, y = clf
    yb = (y > 0).astype(int)
    m2 = LogisticRegression().fit(X[:400], yb[:400])
    Pc, Pt = m2.predict_proba(X[400:800]), m2.predict_proba(X[800:])
    a = pc.VennAbers(m2).calibrate(X[400:800], yb[400:800])
    b = pc.VennAbers().calibrate(proba=Pc, y=yb[400:800])
    c = pc.VennAbers().calibrate(scores=Pc[:, 1], y=yb[400:800])
    np.testing.assert_array_equal(a.predict_proba(X[800:]), b.predict_proba(proba=Pt))
    np.testing.assert_array_equal(a.predict_proba(X[800:]), c.predict_proba(scores=Pt[:, 1]))
    np.testing.assert_array_equal(a.predict_p0_p1(X[800:]), c.predict_p0_p1(scores=Pt[:, 1]))
    assert set(a.predict(X[800:])) <= {0, 1}
    # multiclass
    Pc3, Pt3 = model.predict_proba(X[400:800]), model.predict_proba(X[800:])
    a3 = pc.VennAbers(model).calibrate(X[400:800], y[400:800])
    b3 = pc.VennAbers().calibrate(proba=Pc3, y=y[400:800])
    np.testing.assert_array_equal(a3.predict_proba(X[800:]), b3.predict_proba(proba=Pt3))
    np.testing.assert_allclose(b3.predict_proba(proba=Pt3).sum(1), 1.0)
    with pytest.raises(RuntimeError):
        b3.predict_p0_p1(proba=Pt3)


def test_aci_three_routes_and_stepwise(reg):
    model, X, y = reg
    pcal, pte = model.predict(X[300:600]), model.predict(X[600:])
    kw = dict(alpha=0.1, gamma=0.01)
    a = pc.AdaptiveConformalInference(model, **kw).calibrate(X[300:600], y[300:600]).run(X[600:], y[600:])
    b = pc.AdaptiveConformalInference(**kw).calibrate(y_pred=pcal, y=y[300:600]).run(y_pred=pte, y=y[600:])
    c = pc.AdaptiveConformalInference(**kw).calibrate(scores=np.abs(y[300:600] - pcal)).run(scores=np.abs(y[600:] - pte))
    np.testing.assert_array_equal(a.thresholds, b.thresholds)
    np.testing.assert_array_equal(a.thresholds, c.thresholds)
    np.testing.assert_array_equal(a.intervals, b.intervals)
    assert c.intervals is None
    step = pc.AdaptiveConformalInference(**kw).calibrate(y_pred=pcal, y=y[300:600])
    for t in range(300):
        np.testing.assert_array_equal(step.predict_interval(y_pred=pte[t : t + 1])[0], b.intervals[t])
        assert step.update(y_pred=pte[t : t + 1], y=y[600 + t : 601 + t]) == bool(b.errors[t])
    empty = pc.AdaptiveConformalInference().calibrate()
    assert np.isinf(empty.threshold())


def test_params_follow_sklearn_conventions():
    cp = pc.SplitConformalClassifier(alpha=0.2, score="aps", random_state=1)
    params = cp.get_params()
    assert params["alpha"] == 0.2 and params["estimator"] is None and params["random_state"] == 1
    assert cp.set_params(alpha=0.05).alpha == 0.05
    with pytest.raises(ValueError):
        cp.set_params(nope=1)
