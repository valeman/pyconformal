import math

import numpy as np
import pytest

import pyconformal as pc


def history(rng, T=400, H=4, drift=0.0):
    """AR(1) series with naive-style h-step forecasts; F[t, h-1] is the
    forecast for time t made h steps earlier."""
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * y[t - 1] + rng.normal() * (1.0 + drift * t / T)
    F = np.full((T, H), np.nan)
    for h in range(1, H + 1):
        F[h:, h - 1] = 0.8**h * y[:-h]
    return F, y


def by_definition(F, y, alpha, symmetric, window, min_cal):
    T, H = F.shape
    E = y[:, None] - F
    out = np.full((T, H, 2), np.nan)
    for h in range(1, H + 1):
        for t in range(T):
            if np.isnan(F[t, h - 1]):
                continue
            e = E[: max(t - h + 1, 0), h - 1]  # targets up to t - h
            e = e[~np.isnan(e)]
            if window:
                e = e[-window:]
            if len(e) < min_cal:
                continue
            if symmetric:
                lo = up = pc.conformal_quantile(np.abs(e), alpha)
            else:
                lo, up = pc.conformal_quantile(-e, alpha / 2), pc.conformal_quantile(e, alpha / 2)
            out[t, h - 1] = F[t, h - 1] - lo, F[t, h - 1] + up
    return out


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("window", [None, 40])
def test_matches_definition(symmetric, window):
    rng = np.random.default_rng(0)
    F, y = history(rng)
    y[50:80] = np.round(y[50:80], 1)  # ties
    for alpha in (0.1, 0.2):
        ms = pc.MultiStepSplitConformal(alpha=alpha, symmetric=symmetric, window=window, min_calibration=15)
        got = ms.run(y_pred=F, y=y).intervals
        np.testing.assert_array_equal(got, by_definition(F, y, alpha, symmetric, window, 15))
    # the next H steps use every error seen
    nxt = ms.predict_interval(y_pred=np.zeros(4))
    E = y[:, None] - F
    for h in range(4):
        e = E[:, h][~np.isnan(E[:, h])][-window:] if window else E[:, h][~np.isnan(E[:, h])]
        want = (pc.conformal_quantile(np.abs(e), 0.2),) * 2 if symmetric else (pc.conformal_quantile(-e, 0.1), pc.conformal_quantile(e, 0.1))
        np.testing.assert_allclose(nxt[h], [-want[0], want[1]])


def test_no_look_ahead():
    rng = np.random.default_rng(1)
    F, y = history(rng, T=300, H=5)
    base = pc.MultiStepSplitConformal().run(y_pred=F, y=y).intervals
    y2 = y.copy()
    y2[200:] += 100.0  # change the future
    F2 = F.copy()
    after = pc.MultiStepSplitConformal().run(y_pred=F2, y=y2).intervals
    for h in range(1, 6):
        # intervals for targets up to 199 + h cannot depend on outcomes from 200 on
        np.testing.assert_array_equal(base[: 200 + h, h - 1], after[: 200 + h, h - 1])


def test_coverage_per_horizon_and_widths_grow():
    rng = np.random.default_rng(2)
    cov, width = [], []
    for _ in range(40):
        F, y = history(rng, T=900, H=4)
        r = pc.MultiStepSplitConformal(alpha=0.1, min_calibration=50).run(y_pred=F, y=y)
        cov.append(r.coverage)
        width.append(np.nanmean(r.intervals[:, :, 1] - r.intervals[:, :, 0], axis=0))
    cov, width = np.array(cov), np.mean(width, axis=0)
    assert np.all(np.abs(cov.mean(axis=0) - 0.9) < 0.02)
    assert np.all(np.diff(width) > 0)  # further ahead, wider


def test_window_follows_growing_noise():
    rng = np.random.default_rng(3)
    late_all, late_win = [], []
    for _ in range(40):
        F, y = history(rng, T=1500, H=2, drift=4.0)
        a = pc.MultiStepSplitConformal(min_calibration=50).run(y_pred=F, y=y)
        w = pc.MultiStepSplitConformal(min_calibration=50, window=150).run(y_pred=F, y=y)
        late_all.append(1 - a.errors[1000:].mean()); late_win.append(1 - w.errors[1000:].mean())
    assert np.mean(late_win) > np.mean(late_all) + 0.02
    assert abs(np.mean(late_win) - 0.9) < 0.03


def test_errors():
    rng = np.random.default_rng(4)
    F, y = history(rng, T=60, H=3)
    ms = pc.MultiStepSplitConformal(min_calibration=10)
    with pytest.raises(RuntimeError):
        ms.predict_interval(y_pred=np.zeros(3))
    assert ms.calibrate(y_pred=F, y=y).predict_interval(y_pred=np.zeros(3)).shape == (3, 2)
    with pytest.raises(ValueError):
        ms.predict_interval(y_pred=np.zeros(2))
    with pytest.raises(ValueError):
        ms.run(y_pred=F[:, 0], y=y)
    with pytest.raises(TypeError):
        ms.run(F, y)
    short = pc.MultiStepSplitConformal(min_calibration=100).run(y_pred=F, y=y)
    assert np.isnan(short.intervals).all()
    assert pc.MultiStepSplitConformal(window=5).get_params()["window"] == 5


# ---------------------------------------------------------------- AcMCP
from pathlib import Path  # noqa: E402

from pyconformal import _acmcp_kernel as _ak  # noqa: E402

needs_numba = pytest.mark.skipif(not _ak.HAVE_NUMBA, reason="the scorecaster's model fits are too slow without numba")
R_REF = np.load(Path(__file__).parent / "data" / "acmcp_r.npz")


def acmcp_by_definition(F, y, alpha, ncal, lr, KI, Csat):
    """Quantile tracking plus integrator, written out step by step."""
    T, H = F.shape
    E = y[:, None] - F
    out = np.full((T, H, 2), np.nan)
    for h in range(1, H + 1):
        s = int(np.flatnonzero(~np.isnan(E[:, h - 1]))[0])
        for side, sign in ((0, -1.0), (1, 1.0)):
            qs = np.zeros(T + H)
            qts, misses = 0.0, []
            for t in range(s, T):
                e = E[s : t + 1, h - 1]
                misses.append(float(sign * E[t, h - 1] > qs[t]))
                m = len(misses)
                qts += (lr if m <= 1 else lr * (e.max() - e.min())) * (misses[-1] - alpha / 2)
                integ = 0.0
                if m > 1:
                    arg = (sum(misses) - m * alpha / 2) * math.log(m) / (Csat * m)
                    integ = math.inf if arg >= math.pi / 2 else -math.inf if arg <= -math.pi / 2 else KI * math.tan(arg)
                qs[t + h] = qts + integ
                if m >= ncal and t + h < T:
                    out[t + h, h - 1, side] = F[t + h, h - 1] + sign * qs[t + h]
    return out


def test_acmcp_tracking_matches_definition():
    F, y = history(np.random.default_rng(0), T=250, H=3)
    got = pc.AcMCP(alpha=0.2, min_calibration=20, scorecast=False, KI=1.5, Csat=0.8).run(y_pred=F, y=y)
    want = acmcp_by_definition(F, y, 0.2, 20, 0.1, 1.5, 0.8)
    np.testing.assert_allclose(got.intervals, want, rtol=0, atol=1e-10)


def test_acmcp_matches_r_without_scorecaster():
    ac = pc.AcMCP(alpha=0.1, min_calibration=50, scorecast=False, KI=2.0, Csat=1.5)
    got = ac.run(y_pred=R_REF["F"], y=R_REF["y"]).intervals
    np.testing.assert_allclose(got[:, :, 0], R_REF["lo_plain"], atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(got[:, :, 1], R_REF["up_plain"], atol=1e-10, equal_nan=True)


@needs_numba
@pytest.mark.parametrize("name, window", [("sc", None), ("roll", 50)])
def test_acmcp_matches_r_with_scorecaster(name, window):
    """R refits the MA model from scratch with a loose tolerance and can stop
    at a local optimum, so agreement is close but not exact."""
    ac = pc.AcMCP(alpha=0.1, min_calibration=50, window=window, KI=2.0, Csat=1.5)
    got = ac.run(y_pred=R_REF["F"], y=R_REF["y"]).intervals
    for side, key in ((0, "lo_"), (1, "up_")):
        ref = R_REF[key + name]
        assert np.array_equal(np.isnan(ref), np.isnan(got[:, :, side]))
        d = np.abs(ref - got[:, :, side])[~np.isnan(ref)]
        assert np.median(d) < 1e-4 and d.max() < 0.1
        np.testing.assert_allclose(got[:, 0, side], ref[:, 0], atol=1e-10, equal_nan=True)  # h = 1 has no MA model


@needs_numba
def test_ma_mean_is_maximum_likelihood():
    sm = pytest.importorskip("statsmodels.tsa.arima.model")
    rng = np.random.default_rng(3)
    for q, theta in ((1, [0.6]), (3, [0.5, -0.3, 0.2])):
        eps = rng.normal(size=600 + q)
        x = 0.7 + eps[q:] + sum(c * eps[q - j - 1 : -j - 1] for j, c in enumerate(theta))
        th = np.zeros(q)
        mu, f = _ak._ma_fit(x, 0, len(x), q, th, np.eye(q), True)
        ref = sm.ARIMA(x, order=(0, 0, q), trend="c").fit()
        loglik = -0.5 * len(x) * (f + 1.0 + math.log(2 * math.pi))
        assert loglik >= ref.llf - 1e-6
        assert abs(mu - ref.params[0]) < 2e-3
        np.testing.assert_allclose(th, theta, atol=0.15)


@needs_numba
def test_acmcp_scorecast_first_horizon_is_running_mean():
    F, y = history(np.random.default_rng(4), T=200, H=2)
    ac = pc.AcMCP(min_calibration=30)
    ac.run(y_pred=F, y=y)
    e = (y - F[:, 0])[1:]
    for t in (40, 120, 198):  # scorecast for target t + 1 uses errors of targets 1..t
        assert ac.scorecasts_[t + 1, 0] == pytest.approx(e[:t].mean())


@pytest.mark.parametrize("scorecast", [False, pytest.param(True, marks=needs_numba)])
def test_acmcp_does_not_look_ahead(scorecast):
    F, y = history(np.random.default_rng(5), T=220, H=3)
    kw = dict(alpha=0.1, min_calibration=30, scorecast=scorecast, Csat=1.0)
    a = pc.AcMCP(**kw).run(y_pred=F, y=y).intervals
    y2 = y.copy()
    y2[150:] += 25.0
    b = pc.AcMCP(**kw).run(y_pred=F, y=y2).intervals
    for h in range(1, 4):  # the interval for target t at horizon h is fixed at time t - h
        np.testing.assert_array_equal(a[: 150 + h, h - 1], b[: 150 + h, h - 1])
    assert not np.allclose(a[190:], b[190:])


def test_acmcp_long_run_coverage_under_drift():
    F, y = history(np.random.default_rng(6), T=4000, H=3, drift=4.0)
    res = pc.AcMCP(alpha=0.1, min_calibration=50, scorecast=False).run(y_pred=F, y=y)
    assert np.all(np.abs(res.coverage - 0.9) < 0.02)


def test_acmcp_next_steps_and_errors():
    F, y = history(np.random.default_rng(7), T=150, H=3)
    ac = pc.AcMCP(min_calibration=20, scorecast=False)
    with pytest.raises(RuntimeError):
        ac.predict_interval(y_pred=np.zeros(3))
    ac.run(y_pred=F, y=y)
    nxt = ac.predict_interval(y_pred=np.array([1.0, 2.0, 3.0]))
    assert nxt.shape == (3, 2) and np.all(nxt[:, 0] < nxt[:, 1])
    with pytest.raises(ValueError):
        ac.predict_interval(y_pred=np.zeros(2))
    holes = F.copy()
    holes[80, 1] = np.nan
    with pytest.raises(ValueError):
        ac.run(y_pred=holes, y=y)
    with pytest.raises(ValueError):
        pc.AcMCP(min_calibration=1)
