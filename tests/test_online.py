import math

import numpy as np
import pytest

from pyconformal import AdaptiveConformalInference, conformal_quantile
from pyconformal._online_kernel import HAVE_NUMBA

pytestmark = pytest.mark.filterwarnings("error")
KERNELS = [False] + ([True] if HAVE_NUMBA else [])


def reference_aci(cal, stream, alpha, gamma, window=None):
    """Gibbs and Candes (2021), written as plainly as possible."""
    cal = list(cal)
    if window:
        cal = cal[-window:]
    a = alpha
    qs, errs, alphas = [], [], []
    for s in stream:
        n = len(cal)
        alphas.append(a)
        if a <= 0:
            q = math.inf
        else:
            k = math.ceil((n + 1) * (1 - a) - 1e-9)
            q = math.inf if k > n else (-math.inf if k < 1 else sorted(cal)[k - 1])
        err = s > q
        qs.append(q)
        errs.append(err)
        a = a + gamma * (alpha - err)
        cal.append(s)
        if window:
            cal = cal[-window:]
    return np.array(qs), np.array(errs), np.array(alphas), a


@pytest.mark.parametrize("use_numba", KERNELS)
@pytest.mark.parametrize("window", [None, 1, 7, 50])
@pytest.mark.parametrize("gamma", [0.0, 0.005, 0.05, 0.5])
def test_run_matches_reference(use_numba, window, gamma):
    rng = np.random.default_rng(0)
    for n0 in (0, 3, 40, 120):
        cal = np.round(rng.exponential(size=n0), 1)  # rounding creates ties
        stream = np.round(rng.exponential(size=300) * np.linspace(1, 4, 300), 1)
        q, err, al, final = reference_aci(cal, stream, 0.1, gamma, window)
        m = AdaptiveConformalInference(0.1, gamma, window).fit(cal)
        res = m.run(stream, use_numba=use_numba)
        assert np.array_equal(res.thresholds, q)
        assert np.array_equal(res.errors, err)
        assert np.array_equal(res.alphas, al)
        assert m.alpha_t_ == final


@pytest.mark.parametrize("window", [None, 25])
def test_run_equals_stepwise_updates_and_state(window):
    rng = np.random.default_rng(1)
    cal, stream = rng.exponential(size=30), rng.exponential(size=200)
    a = AdaptiveConformalInference(0.2, 0.03, window).fit(cal)
    b = AdaptiveConformalInference(0.2, 0.03, window).fit(cal)
    res = a.run(stream[:120])
    qs, errs = [], []
    for s in stream[:120]:
        qs.append(b.threshold())
        errs.append(b.update(s))
    assert np.array_equal(res.thresholds, qs) and np.array_equal(res.errors, errs)
    assert a.alpha_t_ == b.alpha_t_
    assert np.array_equal(a.calibration_scores_, b.calibration_scores_)
    # continuing after run() is the same as one long run
    c = AdaptiveConformalInference(0.2, 0.03, window).fit(cal)
    full = c.run(stream)
    rest = a.run(stream[120:])
    assert np.array_equal(full.thresholds[120:], rest.thresholds)


def test_gamma_zero_is_online_split_conformal():
    rng = np.random.default_rng(2)
    cal, stream = rng.normal(size=19), rng.normal(size=50)
    res = AdaptiveConformalInference(0.05, 0.0).fit(cal).run(stream)
    for t in range(50):
        expected = conformal_quantile(np.concatenate([cal, stream[:t]]), 0.05)
        assert res.thresholds[t] == expected
    # n = 19, alpha = 0.05 is the classic floating-point trap: rank must be 19
    assert res.thresholds[0] == np.sort(cal)[18]


@pytest.mark.parametrize("gamma", [0.005, 0.05])
def test_long_run_guarantee_under_distribution_shift(gamma):
    # Adversarial stream: scale jumps, trend and a heavy tail. The bound of
    # Gibbs and Candes holds for any sequence.
    rng = np.random.default_rng(3)
    T, alpha = 20000, 0.1
    scale = np.where((np.arange(T) // 2500) % 2 == 0, 1.0, 8.0) * np.linspace(1, 3, T)
    stream = np.abs(rng.standard_t(2, size=T)) * scale
    m = AdaptiveConformalInference(alpha, gamma, window=200).fit(stream[:0])
    res = m.run(stream)
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(res.errors.mean() - alpha) <= bound + 1e-12
    # plain split conformal (gamma = 0) has no such protection here
    assert res.coverage == pytest.approx(0.9, abs=bound + 1e-12)


def test_iid_coverage():
    rng = np.random.default_rng(4)
    covs = [
        AdaptiveConformalInference(0.1, 0.005).fit(rng.exponential(size=200)).run(
            rng.exponential(size=2000)
        ).coverage
        for _ in range(100)
    ]
    assert 0.895 <= np.mean(covs) <= 0.905


def test_intervals_and_edge_cases():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(size=500))
    pred = np.concatenate([[0.0], y[:-1]])
    m = AdaptiveConformalInference(0.1, 0.01)
    res = m.run_intervals(pred, y)
    iv = res.intervals
    assert iv.shape == (500, 2) and np.all(iv[:, 0] <= iv[:, 1])
    assert math.isinf(iv[0, 1])  # no calibration data yet: infinite, not narrow
    covered = (iv[:, 0] <= y) & (y <= iv[:, 1])
    assert np.array_equal(covered, ~res.errors)
    nxt = m.predict_interval(1.5)
    assert nxt.shape == (2,) and nxt[0] <= 1.5 <= nxt[1]
    # huge values do not overflow or produce NaN
    big = AdaptiveConformalInference(0.1, 0.05).run(np.array([1e20, 1e-20, 3e19, 0.0] * 50))
    assert not np.any(np.isnan(big.thresholds))
    # empty stream
    empty = AdaptiveConformalInference().fit([1.0, 2.0]).run([])
    assert empty.thresholds.shape == (0,)


def test_validation():
    with pytest.raises(ValueError):
        AdaptiveConformalInference(alpha=0)
    with pytest.raises(ValueError):
        AdaptiveConformalInference(gamma=-0.1)
    with pytest.raises(ValueError):
        AdaptiveConformalInference(window=0)
    with pytest.raises(ValueError):
        AdaptiveConformalInference().run([1.0, np.nan])
    with pytest.raises(ValueError):
        AdaptiveConformalInference().run_intervals([1.0, 2.0], [1.0])
