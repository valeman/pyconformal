"""Regression tests for the defects found by the forensic audit of 0.1.0
(November 2026). Each test is one of the audit's counterexamples. The online
ones (ACI, PID, DtACI, rank boundaries) are in test_online.py."""

import numpy as np
import pyconformal as pc
import pytest

pytestmark = pytest.mark.filterwarnings("error")


def _inside(iv, y):
    return (iv[:, 0] <= y) & (y <= iv[:, 1])


class _Mean:
    """Predicts the (weighted) mean of its training targets."""

    def fit(self, X, y, sample_weight=None):
        self.m = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        return np.full(len(X), self.m)


class _Memorizer:
    """Exact on rows it was trained on, zero elsewhere."""

    def fit(self, X, y):
        self.lookup = dict(zip(X[:, 0].tolist(), y.tolist()))
        return self

    def predict(self, X):
        return np.array([self.lookup.get(x, 0.0) for x in X[:, 0].tolist()])


# ------------------------------------------------------------- predictive


@pytest.mark.parametrize("offset", [0.0, 1e6, 1e14])
def test_crps_is_translation_invariant_and_nonnegative(offset):
    y = np.tile([0.0, 2.0, 4.0, 6.0], 250)
    cps = pc.ConformalPredictiveSystem().calibrate(y=y + offset, y_pred=np.zeros(len(y)))
    got = cps.crps(y=[2.0 + offset], y_pred=[0.0])[0]
    assert got == pytest.approx(0.75, abs=1e-9)


def test_crps_matches_direct_formula_with_ties_and_scale():
    rng = np.random.default_rng(1)
    a = np.round(rng.normal(size=200), 1)
    cps = pc.ConformalPredictiveSystem().calibrate(y=a * 3.0, y_pred=np.zeros(200), sigma=np.full(200, 3.0))
    for z in (-5.0, -0.3, 0.0, 0.1, 2.0):
        direct = np.mean(np.abs(a - z)) - 0.5 * np.mean(np.abs(a[:, None] - a[None, :]))
        assert cps.crps(y=[z * 2.0], y_pred=[0.0], sigma=[2.0])[0] == pytest.approx(2.0 * direct, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("rows", [1, 100_000])
def test_missing_cdf_input_is_nan_not_a_tail_probability(rows):
    cps = pc.ConformalPredictiveSystem().calibrate(y=np.arange(9.0), y_pred=np.zeros(9))
    out = cps.predict_cdf(y=np.nan, y_pred=np.zeros(rows))
    assert np.isnan(out).all()
    assert cps.predict_cdf(y=np.inf, y_pred=[0.0])[0] == pytest.approx(0.95)


# ---------------------------------------------------------- weighted split


def test_weighted_mass_boundary_is_not_lowered():
    s = np.arange(1.0, 10.0)
    m = pc.SplitConformalRegressor().calibrate(y=s, y_pred=np.zeros(9), weights=np.ones(9))
    # equal weights: exactly split conformal
    np.testing.assert_array_equal(m.predict_interval(y_pred=[0.0], weights=[1.0]), [[-9.0, 9.0]])
    # a test weight a hair above 1 needs more than 9 units of mass: infinite
    assert np.isinf(m.predict_interval(y_pred=[0.0], weights=[1 + 1e-12])).all()


def test_weighted_rejects_all_zero_and_survives_huge_weights():
    with pytest.raises(ValueError):
        pc.SplitConformalRegressor().calibrate(y=[1.0, 2.0], y_pred=[0.0, 0.0], weights=[0.0, 0.0])
    s = np.arange(1.0, 20.0)
    big = pc.SplitConformalRegressor().calibrate(y=s, y_pred=np.zeros(19), weights=np.full(19, 1e307))
    one = pc.SplitConformalRegressor().calibrate(y=s, y_pred=np.zeros(19), weights=np.ones(19))
    np.testing.assert_array_equal(big.predict_interval(y_pred=[0.0], weights=[1e307]),
                                  one.predict_interval(y_pred=[0.0], weights=[1.0]))


# ------------------------------------------------------------- lifecycle


def test_refit_clears_the_calibration():
    from sklearn.dummy import DummyRegressor

    X = np.arange(19.0)[:, None]
    m = pc.SplitConformalRegressor(DummyRegressor()).fit(X, np.zeros(19)).calibrate(X, np.zeros(19))
    m.fit(X, np.full(19, 100.0))
    with pytest.raises(RuntimeError):
        m.predict_interval(X[:1])
    m.calibrate(X, np.full(19, 100.0))
    np.testing.assert_array_equal(m.predict_interval(X[:1]), [[100.0, 100.0]])


def test_set_params_validates_before_changing_anything():
    c = pc.SplitConformalClassifier().calibrate(proba=np.tile([0.6, 0.4], (19, 1)), y=np.zeros(19, dtype=int))
    with pytest.raises(ValueError):
        c.set_params(score="no-such-score")
    assert c.calibration_scores_ is not None  # unchanged
    c.set_params(alpha=0.2)  # a new level keeps the calibration
    assert c.calibration_scores_ is not None and c.alpha == 0.2
    c.set_params(score="aps")  # a new score is built properly and clears it
    assert isinstance(c.score, pc.APS) and c.calibration_scores_ is None


# -------------------------------------------------------- classification


def test_classifier_rejects_a_different_number_of_columns():
    c = pc.SplitConformalClassifier().calibrate(proba=np.tile([0.6, 0.4], (19, 1)), y=np.zeros(19, dtype=int))
    with pytest.raises(ValueError):
        c.predict_set(proba=[[0.1, 0.1, 0.8]])
    with pytest.raises(ValueError):
        c.predict_set(proba=[[np.nan, 1.0]])
    with pytest.raises(ValueError):
        pc.SplitConformalClassifier(classes=[0, 0]).calibrate(proba=[[0.5, 0.5]], y=[0])


def test_classifier_thresholds_are_cached_and_unchanged():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(4), 5000)
    y = rng.integers(4, size=5000)
    for cond in (False, True):
        c = pc.SplitConformalClassifier(class_conditional=cond).calibrate(proba=P, y=y)
        first = c.thresholds()
        np.testing.assert_array_equal(c.thresholds(), first)
        if cond:
            for k in range(4):
                assert first[k] == pc.conformal_quantile(c.calibration_scores_[y == k], 0.1)
        else:
            assert first == pc.conformal_quantile(c.calibration_scores_, 0.1)


@pytest.mark.parametrize("dtype", [bool, int, str])
def test_venn_abers_respects_class_order_for_every_label_type(dtype):
    labels = {bool: [True, False], int: [1, 0], str: ["b", "a"]}[dtype]
    y = np.array([labels[0], labels[0], labels[1], labels[1]], dtype=dtype if dtype is not str else object)
    va = pc.VennAbers(classes=np.array(labels, dtype=object if dtype is str else dtype)).calibrate(
        scores=[0.0, 0.0, 1.0, 1.0], y=y)
    # the positive class is labels[1], which sits at the high scores
    np.testing.assert_allclose(va.predict_proba(scores=[0.0, 1.0]), [[0.75, 0.25], [0.25, 0.75]])


# --------------------------------------------------------------- ensembles


def test_enbpi_uses_out_of_bag_residuals_only():
    rng = np.random.default_rng(2147)
    X = rng.random((300, 1))
    y = rng.normal(size=300)
    m = pc.EnbPI(_Memorizer(), n_bootstraps=2, random_state=0).fit(X, y)
    assert m.n_starved_ == 0 and m.n_members_ > 2  # extra members were trained
    assert np.all(m.residuals_ > 0)  # no in-bag (zero) residual
    np.testing.assert_allclose(np.sort(m.residuals_), np.sort(np.abs(y)))


def test_enbpi_calibrate_drops_rows_without_out_of_bag_and_spci_refuses():
    P = np.zeros((10, 2))
    bag = np.zeros((10, 2), dtype=bool)
    bag[:3] = True
    m = pc.EnbPI().calibrate(y=np.arange(10.0), y_pred=P, in_bag=bag)
    assert m.n_starved_ == 3 and m.residuals_.shape[0] == 7
    with pytest.raises(ValueError):
        pc.SPCI().calibrate(y=np.arange(10.0), y_pred=P, in_bag=bag)


def test_enbpi_resamples_sample_weight_with_the_rows():
    X = np.arange(20.0)[:, None]
    w = 1.0 + X[:, 0]
    m = pc.EnbPI(_Mean(), n_bootstraps=2, random_state=1).fit(X, X[:, 0], sample_weight=w)
    idx = np.random.default_rng(1).integers(0, 20, size=20)
    assert m.estimators_[0].m == pytest.approx(np.average(X[idx, 0], weights=w[idx]))


def test_cross_conformal_resamples_sample_weight_with_the_rows():
    X = np.arange(20.0)[:, None]
    w = 1.0 + X[:, 0]
    m = pc.CrossConformalRegressor(_Mean(), cv=4).fit(X, X[:, 0], sample_weight=w)
    assert len(m.estimators_) == 4  # a wrong-length weight vector would have failed


@pytest.mark.parametrize("cut", [1, 2, 3, 5, 7])
def test_enbpi_batches_carry_across_run_calls(cut):
    s = np.arange(1.0, 11.0)
    make = lambda: pc.EnbPI(alpha=0.2, batch_size=4).calibrate(y=s, y_pred=np.zeros((10, 2)),
                                                               in_bag=np.zeros((10, 2), bool))
    stream, P = np.full(8, 100.0), np.zeros((8, 2))
    whole = make().run(y=stream, y_pred=P).intervals
    m = make()
    parts = np.vstack([m.run(y=stream[:cut], y_pred=P[:cut]).intervals, m.run(y=stream[cut:], y_pred=P[cut:]).intervals])
    np.testing.assert_array_equal(whole, parts)


# --------------------------------------------------------------- quantiles


def test_cir_shortest_window_warns_and_greedy_covers():
    grid = np.tile([0.0, 1.0, 11.0, 13.0, 15.0], (19, 1))
    y = np.full(19, 2.0)
    greedy = pc.ConformalInterquantileRegressor().calibrate(quantiles=grid, y=y)
    assert _inside(greedy.predict_interval(quantiles=grid[:1]), 2.0).all()
    with pytest.warns(UserWarning, match="no coverage guarantee"):
        pc.ConformalInterquantileRegressor(window="shortest")


# -------------------------------------------------------------- multistep


def test_acmcp_with_fixed_csat_does_not_depend_on_history_length():
    a = np.random.default_rng(3).normal(size=90)
    run = lambda T: pc.AcMCP(scorecast=False, Csat=1.0).run(y=a[:T], y_pred=np.zeros((T, 1))).intervals[:50]
    np.testing.assert_array_equal(run(50), run(90))
