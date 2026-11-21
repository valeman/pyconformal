import math

import numpy as np
import pyconformal as pc
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
        m = AdaptiveConformalInference(None, 0.1, gamma, window).calibrate(scores=cal)
        res = m.run(scores=stream, use_numba=use_numba)
        assert np.array_equal(res.thresholds, q)
        assert np.array_equal(res.errors, err)
        assert np.array_equal(res.alphas, al)
        assert m.alpha_t_ == final


@pytest.mark.parametrize("window", [None, 25])
def test_run_equals_stepwise_updates_and_state(window):
    rng = np.random.default_rng(1)
    cal, stream = rng.exponential(size=30), rng.exponential(size=200)
    a = AdaptiveConformalInference(None, 0.2, 0.03, window).calibrate(scores=cal)
    b = AdaptiveConformalInference(None, 0.2, 0.03, window).calibrate(scores=cal)
    res = a.run(scores=stream[:120])
    qs, errs = [], []
    for s in stream[:120]:
        qs.append(b.threshold())
        errs.append(b.update(score=s))
    assert np.array_equal(res.thresholds, qs) and np.array_equal(res.errors, errs)
    assert a.alpha_t_ == b.alpha_t_
    assert np.array_equal(a.calibration_scores_, b.calibration_scores_)
    # continuing after run() is the same as one long run
    c = AdaptiveConformalInference(None, 0.2, 0.03, window).calibrate(scores=cal)
    full = c.run(scores=stream)
    rest = a.run(scores=stream[120:])
    assert np.array_equal(full.thresholds[120:], rest.thresholds)


def test_gamma_zero_is_online_split_conformal():
    rng = np.random.default_rng(2)
    cal, stream = rng.normal(size=19), rng.normal(size=50)
    res = AdaptiveConformalInference(None, 0.05, 0.0).calibrate(scores=cal).run(scores=stream)
    for t in range(50):
        expected = conformal_quantile(np.concatenate([cal, stream[:t]]), 0.05)
        assert res.thresholds[t] == expected
    # n = 19, alpha = 0.05: (n + 1)(1 - alpha) is exactly 19, so the rank must be 19
    assert res.thresholds[0] == np.sort(cal)[18]


@pytest.mark.parametrize("gamma", [0.005, 0.05])
def test_long_run_guarantee_under_distribution_shift(gamma):
    # Adversarial stream: scale jumps, trend and a heavy tail. The bound of
    # Gibbs and Candes holds for any sequence.
    rng = np.random.default_rng(3)
    T, alpha = 20000, 0.1
    scale = np.where((np.arange(T) // 2500) % 2 == 0, 1.0, 8.0) * np.linspace(1, 3, T)
    stream = np.abs(rng.standard_t(2, size=T)) * scale
    m = AdaptiveConformalInference(None, alpha, gamma, window=200).calibrate(scores=stream[:0])
    res = m.run(scores=stream)
    bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
    assert abs(res.errors.mean() - alpha) <= bound + 1e-12
    # plain split conformal (gamma = 0) has no such protection here
    assert res.coverage == pytest.approx(0.9, abs=bound + 1e-12)


def test_iid_coverage():
    rng = np.random.default_rng(4)
    covs = [
        AdaptiveConformalInference(None, 0.1, 0.005).calibrate(scores=rng.exponential(size=200)).run(
            scores=rng.exponential(size=2000)
        ).coverage
        for _ in range(100)
    ]
    assert 0.895 <= np.mean(covs) <= 0.905


def test_intervals_and_edge_cases():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(size=500))
    pred = np.concatenate([[0.0], y[:-1]])
    m = AdaptiveConformalInference(None, 0.1, 0.01)
    res = m.run(y_pred=pred, y=y)
    iv = res.intervals
    assert iv.shape == (500, 2) and np.all(iv[:, 0] <= iv[:, 1])
    assert math.isinf(iv[0, 1])  # no calibration data yet: infinite, not narrow
    covered = (iv[:, 0] <= y) & (y <= iv[:, 1])
    assert np.array_equal(covered, ~res.errors)
    nxt = m.predict_interval(y_pred=1.5)
    assert nxt.shape == (1, 2) and nxt[0, 0] <= 1.5 <= nxt[0, 1]
    # huge values do not overflow or produce NaN
    big = AdaptiveConformalInference(None, 0.1, 0.05).run(scores=np.array([1e20, 1e-20, 3e19, 0.0] * 50))
    assert not np.any(np.isnan(big.thresholds))
    # empty stream
    empty = AdaptiveConformalInference().calibrate(scores=[1.0, 2.0]).run(scores=[])
    assert empty.thresholds.shape == (0,)


def test_validation():
    with pytest.raises(ValueError):
        AdaptiveConformalInference(alpha=0)
    with pytest.raises(ValueError):
        AdaptiveConformalInference(gamma=-0.1)
    with pytest.raises(ValueError):
        AdaptiveConformalInference(window=0)
    with pytest.raises(ValueError):
        AdaptiveConformalInference().run(scores=[1.0, np.nan])
    with pytest.raises(ValueError):
        AdaptiveConformalInference().run(y_pred=[1.0, 2.0], y=[1.0])


# ---------------------------------------------------------------- DtACI


def drifting(rng, T):
    """Scores whose scale jumps, so a fixed calibration set is wrong."""
    scale = np.where((np.arange(T) // 400) % 2 == 0, 1.0, 4.0)
    return np.abs(rng.normal(size=T)) * scale


@pytest.mark.parametrize("window", [None, 150])
def test_dtaci_with_one_step_size_is_aci(window):
    rng = np.random.default_rng(0)
    s = drifting(rng, 1500)
    a = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.02, window=window).calibrate(scores=s[:100]).run(scores=s[100:])
    d = pc.DtACI(alpha=0.1, gammas=[0.02], window=window).calibrate(scores=s[:100]).run(scores=s[100:])
    np.testing.assert_array_equal(a.thresholds, d.thresholds)
    np.testing.assert_array_equal(a.errors, d.errors)
    np.testing.assert_allclose(a.alphas, d.alphas, rtol=0, atol=1e-12)


@pytest.mark.parametrize("window", [None, 200])
def test_dtaci_run_equals_stepwise_and_paths_agree(window):
    rng = np.random.default_rng(1)
    s = drifting(rng, 900)
    s[200:260] = np.round(s[200:260], 1)  # ties
    fast = pc.DtACI(alpha=0.1, window=window).calibrate(scores=s[:50])
    res = fast.run(scores=s[50:])
    slow = pc.DtACI(alpha=0.1, window=window).calibrate(scores=s[:50])
    res_py = slow.run(scores=s[50:], use_numba=False)
    np.testing.assert_array_equal(res.thresholds, res_py.thresholds)
    np.testing.assert_array_equal(res.alphas, res_py.alphas)
    step = pc.DtACI(alpha=0.1, window=window).calibrate(scores=s[:50])
    for t, v in enumerate(s[50:]):
        assert step.threshold() == res.thresholds[t]
        assert step.update(score=v) == bool(res.errors[t])
    assert step.alpha_t_ == fast.alpha_t_
    np.testing.assert_array_equal(step.weights_, fast.weights_)
    np.testing.assert_array_equal(step.expert_alphas_, fast.expert_alphas_)
    # a second stretch continues from the same state
    more = drifting(rng, 100)
    np.testing.assert_array_equal(fast.run(scores=more).thresholds, slow.run(scores=more, use_numba=False).thresholds)


def test_dtaci_tracks_drift_without_choosing_a_step_size():
    rng = np.random.default_rng(2)
    cov = {"dtaci": [], "slow": [], "split": []}
    worst = {"dtaci": [], "slow": []}
    for _ in range(40):
        s = drifting(rng, 4100)
        d = pc.DtACI(alpha=0.1).calibrate(scores=s[:100]).run(scores=s[100:])
        a = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.0005).calibrate(scores=s[:100]).run(scores=s[100:])
        f = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.0).calibrate(scores=s[:100]).run(scores=s[100:])
        cov["dtaci"].append(d.coverage); cov["slow"].append(a.coverage); cov["split"].append(f.coverage)
        block = lambda r: min((1 - r.errors[i : i + 400].mean()) for i in range(300, 3700, 400))
        worst["dtaci"].append(block(d)); worst["slow"].append(block(a))
    assert abs(np.mean(cov["dtaci"]) - 0.9) < 0.01
    assert np.mean(worst["dtaci"]) > np.mean(worst["slow"]) + 0.03  # holds up inside the noisy stretches
    assert np.mean(worst["dtaci"]) > 0.84
    d = pc.DtACI(alpha=0.1)
    assert d.get_params()["interval"] == 500 and len(d.gammas) == 8
    with pytest.raises(ValueError):
        pc.DtACI(gammas=[])
    res = pc.DtACI().calibrate(y_pred=np.zeros(50), y=rng.normal(size=50)).run(y_pred=np.zeros(200), y=rng.normal(size=200))
    assert res.intervals.shape == (200, 2)


# ---------------------------------------------------------------- NexCP


def test_nexcp_matches_definition_stepwise_and_paths():
    rng = np.random.default_rng(3)
    s = drifting(rng, 700)
    s[100:140] = np.round(s[100:140], 1)
    for rho in (0.9, 0.99, 1.0):
        m = pc.NonexchangeableConformal(alpha=0.1, rho=rho).calibrate(scores=s[:40])
        res = m.run(scores=s[40:])
        py = pc.NonexchangeableConformal(alpha=0.1, rho=rho).calibrate(scores=s[:40]).run(scores=s[40:], use_numba=False)
        np.testing.assert_array_equal(res.thresholds, py.thresholds)
        step = pc.NonexchangeableConformal(alpha=0.1, rho=rho).calibrate(scores=s[:40])
        for t in range(660):
            past = s[: 40 + t]
            w = rho ** np.arange(len(past), 0, -1.0)
            o = np.argsort(past, kind="stable")
            cum = np.cumsum(np.append(w[o], 1.0)) / (w.sum() + 1.0)
            want = np.append(past[o], np.inf)[np.searchsorted(cum, 0.9 * (1 - 1e-12))]
            assert res.thresholds[t] == want
            assert step.threshold() == want
            step.update(score=s[40 + t])
    # rho = 1 is split conformal on everything seen so far
    plain = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.0).calibrate(scores=s[:40]).run(scores=s[40:])
    np.testing.assert_array_equal(res.thresholds, plain.thresholds)


def test_nexcp_long_stream_does_not_overflow_and_adapts():
    rng = np.random.default_rng(4)
    s = drifting(rng, 60_000)
    res = pc.NonexchangeableConformal(alpha=0.1, rho=0.98).run(scores=s)  # weights would reach e^1200 unscaled
    assert np.all(np.isfinite(res.thresholds[200:]))
    flat = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.0).run(scores=s)
    block = lambda r: min((1 - r.errors[i : i + 400].mean()) for i in range(2100, 59000, 400))
    assert block(res) > block(flat) + 0.04  # unweighted conformal collapses in the noisy stretches
    assert abs(res.coverage - 0.9) < 0.02
    with pytest.raises(ValueError):
        pc.NonexchangeableConformal(rho=0.0)
    iv = pc.NonexchangeableConformal().calibrate(y_pred=np.zeros(30), y=rng.normal(size=30)).predict_interval(y_pred=[1.0])
    assert iv.shape == (1, 2)


# ------------------------------------------------------- AgACI and PID


@pytest.mark.parametrize("window", [None, 200])
def test_agaci_run_equals_stepwise_and_paths_agree(window):
    rng = np.random.default_rng(5)
    s = drifting(rng, 800)
    s[300:340] = np.round(s[300:340], 1)
    for n0 in (0, 60):
        fast = pc.AgACI(alpha=0.1, window=window).calibrate(scores=s[:n0])
        res = fast.run(scores=s[n0:])
        py = pc.AgACI(alpha=0.1, window=window).calibrate(scores=s[:n0]).run(scores=s[n0:], use_numba=False)
        np.testing.assert_array_equal(res.thresholds, py.thresholds)
        step = pc.AgACI(alpha=0.1, window=window).calibrate(scores=s[:n0])
        for t, v in enumerate(s[n0:]):
            assert step.threshold() == res.thresholds[t], t
            assert step.update(score=v) == bool(res.errors[t])
        np.testing.assert_array_equal(step.expert_alphas_, fast.expert_alphas_)
        more = drifting(rng, 50)  # a second stretch continues from the same state
        again = fast.run(scores=more)
        for t, v in enumerate(more):
            assert step.threshold() == again.thresholds[t]
            step.update(score=v)


def test_agaci_single_expert_and_behaviour():
    rng = np.random.default_rng(6)
    s = drifting(rng, 3000)
    # one expert: its threshold, with infinite values replaced by the largest score so far
    one = pc.AgACI(alpha=0.1, gammas=[0.01]).calibrate(scores=s[:100]).run(scores=s[100:])
    aci = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.01).calibrate(scores=s[:100]).run(scores=s[100:])
    cap = np.maximum.accumulate(s)[99:-1]
    np.testing.assert_allclose(one.thresholds, np.clip(aci.thresholds, 0.0, cap), rtol=0, atol=1e-12)
    cov, worst_ag, worst_fix = [], [], []
    for _ in range(30):
        s = drifting(rng, 4100)
        r = pc.AgACI(alpha=0.1).calibrate(scores=s[:100]).run(scores=s[100:])
        f = pc.AdaptiveConformalInference(alpha=0.1, gamma=0.0).calibrate(scores=s[:100]).run(scores=s[100:])
        block = lambda z: min((1 - z.errors[i : i + 400].mean()) for i in range(300, 3700, 400))
        cov.append(r.coverage); worst_ag.append(block(r)); worst_fix.append(block(f))
        assert np.all(np.isfinite(r.thresholds))
    assert 0.89 < np.mean(cov) < 0.93
    assert np.mean(worst_ag) > np.mean(worst_fix) + 0.05
    assert len(pc.AgACI().gammas) == 30
    assert math.isinf(pc.AgACI().threshold())  # nothing seen yet


def test_pid_matches_definition_and_tracks():
    rng = np.random.default_rng(7)
    s = drifting(rng, 4000)
    m = pc.ConformalPID(alpha=0.1, lr=0.1)
    res = m.run(scores=s)
    q, hist = 0.0, []
    for t, v in enumerate(s):  # quantile tracking, written out
        assert res.thresholds[t] == q
        lr = 0.1 * (max(hist[-100:]) - min(hist[-100:])) if t > 0 else 0.1
        q = q + lr * (0.9 if v > q else -0.1)
        hist.append(v)
    assert abs(res.coverage - 0.9) < 0.015
    assert np.all(np.isfinite(res.thresholds))
    block = min((1 - res.errors[i : i + 400].mean()) for i in range(400, 3600, 400))
    assert block > 0.84
    step = pc.ConformalPID(alpha=0.1, KI=1.0)
    ref = pc.ConformalPID(alpha=0.1, KI=1.0).run(scores=s[:500])
    for t in range(500):
        assert step.threshold() == ref.thresholds[t]
        assert step.update(score=s[t]) == bool(ref.errors[t])
    warm = pc.ConformalPID(alpha=0.1).calibrate(scores=s[:300])
    cold = pc.ConformalPID(alpha=0.1)
    cold.run(scores=s[:300])
    assert warm.threshold() == cold.threshold() and warm.threshold() > 0
    iv = pc.ConformalPID().calibrate(y_pred=np.zeros(200), y=rng.normal(size=200)).run(y_pred=np.zeros(50), y=rng.normal(size=50))
    assert iv.intervals.shape == (50, 2)
    with pytest.raises(ValueError):
        pc.ConformalPID(lr=0.0)
