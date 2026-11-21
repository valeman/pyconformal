import math

import numpy as np
import pytest

import pyconformal as pc
from pyconformal.quantile import _greedy

LEVELS = np.linspace(0.01, 0.99, 50)


def skewed(rng, n, noise=0.1):
    """Skewed outcomes and an imperfect quantile model for them."""
    x = rng.uniform(0.5, 2.0, n)
    y = x * rng.gamma(2.0, 1.0, n)
    base = np.quantile(rng.gamma(2.0, 1.0, 20_000), LEVELS)
    Q = x[:, None] * base[None, :] * np.exp(0.2 * rng.normal(size=(n, 1)))
    return Q + noise * rng.normal(size=Q.shape), y  # noise makes quantiles cross


def covered(iv, y):
    return (iv[:, 0] <= y) & (y <= iv[:, 1])


def bin_of(Q, y):
    Q = np.sort(Q, axis=1)
    b = (Q < y[:, None]).sum(axis=1)
    b[(b == 0) & (y >= Q[:, 0])] = 1
    return b


METHODS = {
    "cqr": lambda: pc.ConformalizedQuantileRegressor(alpha=0.1),
    "cqr_asym": lambda: pc.ConformalizedQuantileRegressor(alpha=0.1, symmetric=False),
    "cir": lambda: pc.ConformalInterquantileRegressor(alpha=0.1),
    "cir_plus": lambda: pc.ConformalInterquantileRegressor(alpha=0.1, plus=True),
    "cti": lambda: pc.ConformalThresholdedIntervals(alpha=0.1),
}


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("n_cal", [30, 300])
def test_coverage(name, n_cal):
    rng = np.random.default_rng(100 + n_cal)
    cov = []
    for _ in range(300):
        Qc, yc = skewed(rng, n_cal)
        Qt, yt = skewed(rng, 200)
        if name.startswith("cqr"):
            Qc, Qt = Qc[:, [2, -3]], Qt[:, [2, -3]]
        m = METHODS[name]().calibrate(quantiles=Qc, y=yc)
        if name == "cti":
            hit = m.predict_set(quantiles=Qt)[np.arange(200), bin_of(Qt, yt)]
        else:
            hit = covered(m.predict_interval(quantiles=Qt), yt)
        cov.append(hit.mean())
    cov = np.array(cov)
    assert cov.mean() >= 0.9 - 3 * cov.std() / math.sqrt(len(cov))
    assert cov.mean() < 0.96  # valid, and not by being uselessly wide


def test_cir_adapts_to_skewness():
    rng = np.random.default_rng(1)
    Qc, yc = skewed(rng, 2000, noise=0.02)
    Qt, _ = skewed(rng, 4000, noise=0.02)
    cqr = pc.ConformalizedQuantileRegressor().calibrate(quantiles=Qc[:, [2, -3]], y=yc)
    cir = pc.ConformalInterquantileRegressor().calibrate(quantiles=Qc, y=yc)
    plus = pc.ConformalInterquantileRegressor(plus=True).calibrate(quantiles=Qc, y=yc)
    width = lambda iv: (iv[:, 1] - iv[:, 0]).mean()
    w_cqr = width(cqr.predict_interval(quantiles=Qt[:, [2, -3]]))
    w_cir = width(cir.predict_interval(quantiles=Qt))
    w_plus = width(plus.predict_interval(quantiles=Qt))
    assert w_cir < 0.97 * w_cqr
    assert w_plus <= w_cir


def test_greedy_matches_definition():
    rng = np.random.default_rng(2)
    L = rng.exponential(size=(40, 12))
    L[:5] = 1.0  # ties
    step, added, snaps = _greedy(L, (1, 4, 12))
    for i in range(40):
        lo = hi = int(np.argmin(L[i]))
        order, lens = [lo], [L[i, lo]]
        while len(order) < 12:
            left = L[i, lo - 1] if lo > 0 else np.inf
            right = L[i, hi + 1] if hi < 11 else np.inf
            if left <= right:
                lo -= 1
                order.append(lo)
            else:
                hi += 1
                order.append(hi)
            lens.append(min(left, right))
            if len(order) == 4:
                assert (snaps[4][0][i], snaps[4][1][i]) == (lo, hi)
        assert [int(step[i, j]) for j in order] == list(range(1, 13))
        np.testing.assert_array_equal(added[i], lens)
        assert (snaps[12][0][i], snaps[12][1][i]) == (0, 11)


@pytest.mark.parametrize("plus", [False, True])
def test_cir_interval_is_exactly_the_score_region(plus):
    """y is in the interval if and only if its score is at most the threshold."""
    rng = np.random.default_rng(3)
    Qc, yc = skewed(rng, 400)
    m = pc.ConformalInterquantileRegressor(plus=plus).calibrate(quantiles=Qc, y=yc)
    s = m.threshold()
    Qt, yt = skewed(rng, 3000)
    iv = m.predict_interval(quantiles=Qt)
    probe = pc.ConformalInterquantileRegressor(plus=plus).calibrate(quantiles=Qt, y=yt)
    inside = covered(iv, yt)
    # points exactly on a boundary are measure zero; compare the rest
    assert np.array_equal(inside, probe.calibration_scores_ <= s)


def test_cti_set_is_exactly_the_score_region():
    rng = np.random.default_rng(4)
    Qc, yc = skewed(rng, 400)
    m = pc.ConformalThresholdedIntervals().calibrate(quantiles=Qc, y=yc)
    Qt, yt = skewed(rng, 3000)
    sets = m.predict_set(quantiles=Qt)
    probe = pc.ConformalThresholdedIntervals().calibrate(quantiles=Qt, y=yt)
    np.testing.assert_array_equal(sets[np.arange(3000), bin_of(Qt, yt)], probe.calibration_scores_ <= m.threshold())
    size = m.predict_size(quantiles=Qt)
    L = np.diff(np.sort(Qt, axis=1), axis=1)
    np.testing.assert_allclose(size, (L * sets[:, 1:-1]).sum(axis=1))
    hull = m.predict_interval(quantiles=Qt)
    assert np.all((hull[:, 1] - hull[:, 0] >= size - 1e-9) | np.isnan(hull[:, 0]))


def test_cti_finds_two_modes():
    # bimodal outcome: quantiles bunch around -2 and +2
    rng = np.random.default_rng(5)
    def draw(n):
        return np.where(rng.random(n) < 0.5, -2.0, 2.0) + 0.3 * rng.normal(size=n)
    q = np.quantile(draw(50_000), np.linspace(0.005, 0.995, 100))
    Qc = np.tile(q, (500, 1))
    m = pc.ConformalThresholdedIntervals(alpha=0.2).calibrate(quantiles=Qc, y=draw(500))
    kept = m.predict_set(quantiles=q[None, :])[0, 1:-1]
    mids = 0.5 * (q[1:] + q[:-1])
    assert kept[np.abs(np.abs(mids) - 2) < 0.2].all()
    assert not kept[np.abs(mids) < 0.5].any()  # the gap between the modes is left out
    assert m.predict_size(quantiles=q[None, :])[0] < 0.5 * (q[-1] - q[0])


def test_cqr_reference_and_crossing():
    rng = np.random.default_rng(6)
    Qc, yc = skewed(rng, 199)
    Qc = Qc[:, [2, -3]]
    m = pc.ConformalizedQuantileRegressor(alpha=0.1).calibrate(quantiles=Qc, y=yc)
    e = np.sort(np.maximum(Qc[:, 0] - yc, yc - Qc[:, 1]))
    assert m.corrections() == (e[179], e[179])  # ceil(200 * 0.9) = 180th smallest
    iv = m.predict_interval(quantiles=np.array([[1.0, 3.0]]))
    np.testing.assert_allclose(iv, [[1.0 - e[179], 3.0 + e[179]]])
    # quantile model far too wide: the correction is negative and ends may cross
    wide = pc.ConformalizedQuantileRegressor().calibrate(
        quantiles=np.column_stack([np.full(100, -10.0), np.full(100, 10.0)]), y=rng.normal(size=100)
    )
    assert wide.corrections()[0] < 0
    out = wide.predict_interval(quantiles=np.array([[-10.0, 10.0], [0.0, 1.0]]))
    assert out[0, 0] < out[0, 1] and out[1, 0] == out[1, 1] == 0.5
    multi = m.predict_interval(quantiles=Qc, alpha=[0.2, 0.05])
    assert multi.shape == (199, 2, 2) and np.all(multi[:, 1, 1] >= multi[:, 1, 0])


class QuantileModel:
    """Stand-in for a quantile forest: predict(X, quantiles=[...])."""

    def fit(self, X, y):
        self.sd_ = float(np.std(y - X[:, 0]))
        return self

    def predict(self, X, quantiles):
        from statistics import NormalDist
        z = np.array([NormalDist().inv_cdf(min(max(q, 1e-3), 1 - 1e-3)) for q in quantiles])
        return X[:, [0]] + self.sd_ * z[None, :]


class Shift:
    def __init__(self, c):
        self.c = c

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X[:, 0] + self.c


def test_model_routes():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(900, 1))
    y = X[:, 0] + rng.normal(size=900)
    for cls, kw in [
        (pc.ConformalizedQuantileRegressor, {}),
        (pc.ConformalInterquantileRegressor, {"quantile_levels": 20}),
        (pc.ConformalThresholdedIntervals, {"quantile_levels": np.linspace(0.01, 0.99, 50)}),
    ]:
        a = cls(QuantileModel(), **kw).fit(X[:300], y[:300]).calibrate(X[300:600], y[300:600])
        lv = list(a._levels())
        b = cls(**kw).calibrate(quantiles=a.estimator.predict(X[300:600], quantiles=lv), y=y[300:600])
        out_a = a.predict_interval(X[600:])
        np.testing.assert_array_equal(out_a, b.predict_interval(quantiles=a.estimator.predict(X[600:], quantiles=lv)))
        assert 0.84 < covered(out_a, y[600:]).mean() < 0.97
    pair = pc.ConformalizedQuantileRegressor((Shift(-1.5), Shift(1.5))).fit(X, y).calibrate(X[300:600], y[300:600])
    assert 0.84 < covered(pair.predict_interval(X[600:]), y[600:]).mean() < 0.97


def test_edge_cases_and_errors():
    rng = np.random.default_rng(8)
    Q, y = skewed(rng, 5)
    for m in (pc.ConformalInterquantileRegressor(), pc.ConformalInterquantileRegressor(plus=True)):
        iv = m.calibrate(quantiles=Q, y=y).predict_interval(quantiles=Q)  # 5 points cannot give 90%
        assert np.all(np.isinf(iv))
    cti = pc.ConformalThresholdedIntervals().calibrate(quantiles=Q, y=y)
    assert cti.predict_set(quantiles=Q).all() and np.all(np.isinf(cti.predict_size(quantiles=Q)))
    # every calibration outcome beyond the predicted range: infinite, not narrow
    far = pc.ConformalInterquantileRegressor().calibrate(quantiles=np.tile(LEVELS, (50, 1)), y=np.full(50, 99.0))
    assert np.all(np.isinf(far.predict_interval(quantiles=LEVELS[None, :])))
    with pytest.raises(ValueError):
        pc.ConformalizedQuantileRegressor().calibrate(quantiles=Q, y=y)  # needs exactly 2 columns
    with pytest.raises(ValueError):
        pc.ConformalInterquantileRegressor().calibrate(quantiles=Q[:, :1], y=y)
    with pytest.raises(ValueError):
        pc.ConformalInterquantileRegressor(plus=True, window="shortest")
    with pytest.raises(RuntimeError):
        pc.ConformalThresholdedIntervals().predict_set(quantiles=Q)
    with pytest.raises(ValueError):
        cti.predict_set(quantiles=Q[:, :10])  # different number of quantiles
    with pytest.raises(TypeError):
        pc.ConformalInterquantileRegressor().calibrate(Q, y)  # X without a model


@pytest.mark.parametrize("plus", [False, True])
def test_compiled_and_numpy_paths_agree(plus):
    from pyconformal import quantile as M

    if not M.HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(9)
    Qc, yc = skewed(rng, 600)
    Qt, _ = skewed(rng, 800)
    Qt[:20] = np.round(Qt[:20], 0)  # ties and zero-length intervals
    Qc[:20] = np.round(Qc[:20], 0)
    yc[:40] = 1e6 * np.sign(rng.normal(size=40))  # tails

    Qc, yc, Qt = np.tile(Qc, (40, 1)), np.tile(yc, 40), np.tile(Qt, (30, 1))  # large enough for every kernel

    def run():
        m = pc.ConformalInterquantileRegressor(plus=plus).calibrate(quantiles=Qc, y=yc)
        cti = pc.ConformalThresholdedIntervals().calibrate(quantiles=Qc, y=yc)
        cqr = pc.ConformalizedQuantileRegressor().calibrate(quantiles=Qc[:, [3, -4]], y=yc)
        return m.calibration_scores_, [m.predict_interval(quantiles=Qt, alpha=a) for a in (0.5, 0.2, 0.1, 0.99)] + [
            cti.calibration_scores_, cti.predict_set(quantiles=Qt), cqr.calibration_scores_,
            cqr.predict_interval(quantiles=Qt[:, [3, -4]]), cqr.predict_interval(quantiles=Qt[:, [3, -4]], alpha=0.9),
        ]

    fast = run()
    M.HAVE_NUMBA = False
    try:
        slow = run()
    finally:
        M.HAVE_NUMBA = True
    np.testing.assert_array_equal(fast[0], slow[0])
    for f, sl in zip(fast[1], slow[1]):
        np.testing.assert_array_equal(f, sl)


def test_non_finite_quantiles_are_rejected_on_every_path():
    rng = np.random.default_rng(10)
    for n in (50, 30_000):
        Q, y = skewed(rng, n)
        bad = Q.copy()
        bad[n // 2, 7] = np.nan
        for m, cols in ((pc.ConformalInterquantileRegressor(), slice(None)), (pc.ConformalThresholdedIntervals(), slice(None)),
                        (pc.ConformalizedQuantileRegressor(), [2, 7])):
            with pytest.raises(ValueError, match="finite"):
                m.calibrate(quantiles=bad[:, cols], y=y)
            m.calibrate(quantiles=Q[:, cols], y=y)
            with pytest.raises(ValueError, match="finite"):
                (m.predict_set if isinstance(m, pc.ConformalThresholdedIntervals) else m.predict_interval)(quantiles=bad[:, cols])
