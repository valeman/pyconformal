import numpy as np
import pytest

from pyconformal import MultiQT, PanelPredictiveSystem, WeightedTQA


def reference_multiqt(Y, levels, base, lr, delay=0, window=50, floor=0.1):
    """Procedure 1 of Ding, Gibbs and Tibshirani (arXiv:2512.23671) with the step-size rule of their Section 5,
    written step by step for one series. Checked separately against the authors' own code (see benchmarks/multiqt)."""
    def iso(v):
        v = list(v); w = [1.0] * len(v); i = 0
        blocks = []
        for x in v:
            blocks.append([x, 1.0])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                b = blocks.pop(); a = blocks.pop(); blocks.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
        return np.array([b[0] for b in blocks for _ in range(int(b[1]))])
    T, K = len(Y), len(levels); hidden = np.zeros(K); Q = np.empty((T, K)); seen = []
    for t in range(T):
        Q[t] = iso(base[t] + hidden)
        s = t - delay
        if s >= 0:
            seen.append(s)
            if lr == "adaptive":
                recent = [u for u in seen if u >= t + 1 - window]
                eta = max(floor, 0.1 * np.quantile(np.abs(Y[recent][:, None] - base[recent]), 0.9))
            else:
                eta = lr
            hidden = hidden - eta * ((Y[s] <= Q[s]) - np.asarray(levels))
    return Q


@pytest.mark.parametrize("lr,delay", [("adaptive", 0), ("adaptive", 3), (0.3, 0), (0.05, 2)])
def test_matches_reference(lr, delay):
    rng = np.random.default_rng(0)
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    Y = np.cumsum(rng.normal(size=300)) * 0.5
    base = np.r_[0, Y[:-1]][:, None] + np.quantile(rng.normal(size=400), levels)[None]
    got = MultiQT(levels, lr=lr).run(base, Y, delay=delay).quantiles[:, 0]
    assert np.allclose(got, reference_multiqt(Y, levels, base, lr, delay), atol=1e-12)


def test_calibrated_and_ordered_under_drift():
    rng = np.random.default_rng(1)
    levels = np.round(np.arange(0.1, 0.91, 0.1), 1)
    T = 4000
    Y = 0.004 * np.arange(T) + rng.normal(size=T) * (1 + (np.arange(T) > 2000) * 3)      # trend and a jump in noise
    base = np.tile(np.quantile(rng.normal(size=500), levels), (T, 1))                     # a forecaster that never adapts
    res = MultiQT(levels, lr=0.2).run(base, Y)
    assert np.all(np.diff(res.quantiles, axis=2) >= -1e-12)
    assert np.abs(res.frequencies - levels).max() < 0.03
    raw = np.abs((Y[:, None] <= base).mean(0) - levels).max()
    assert raw > 0.3
    assert res.interval(0.2).shape == (T, 1, 2) and 0 <= res.cdf(0.0).min() <= res.cdf(0.0).max() <= 1


def test_many_series_missing_feedback_and_point_forecasts():
    rng = np.random.default_rng(2)
    levels = [0.1, 0.5, 0.9]
    Y = rng.normal(size=(200, 5)).cumsum(0) * 0.3
    point = np.r_[np.zeros((1, 5)), Y[:-1]]
    seen = rng.random((200, 5)) < 0.5
    res = MultiQT(levels).run(point, Y, observed=seen)
    for i in range(5):                                                                    # each series runs on its own
        one = MultiQT(levels).run(point[:, i], Y[:, i], observed=seen[:, i])
        assert np.allclose(res.quantiles[:, i], one.quantiles[:, 0], atol=1e-12)
    never = MultiQT(levels).run(point, Y, observed=np.zeros((200, 5), bool))
    assert np.allclose(never.quantiles, point[:, :, None])                                # no feedback: base returned
    with pytest.raises(ValueError):
        MultiQT([0.5, 0.2])


def test_panel_predictive_system():
    cal, pin, tail = [], [], []
    for r in range(4):
        rng = np.random.default_rng(r)
        T, N, M = 300, 150, 30
        hard = rng.random(N + M) < 0.2
        bias = np.where(rng.random(N + M) < 0.3, 1.0, 0.0)                                # some units are forecast with a bias
        X = np.where(hard, 2.0, 0.0)[None, :, None] + rng.normal(size=(T, N + M, 2))
        Y = bias + rng.normal(size=(T, N + M)) * np.where(hard, 3.0, 1.0)
        P = np.zeros((T, N + M))
        kw = dict(y_pred_cal=P[:, :N], y_cal=Y[:, :N], y_pred_target=P[:, N:], y_target=Y[:, N:], revealed=rng.random(T) < 0.3)
        res = PanelPredictiveSystem().run(X_cal=X[:, :N], X_target=X[:, N:], **kw)
        assert res.quantiles.shape == (T, M, 19) and np.all(np.diff(res.quantiles, axis=2) >= -1e-12)
        base_err = np.abs((Y[:, N:, None] <= res.base_quantiles).mean(0) - res.levels).mean(1)
        cal.append((res.calibration_error.mean(), base_err.mean()))
        d = Y[:, N:, None] - res.base_quantiles
        pin.append((res.pinball.mean(), np.maximum(res.levels * d, (res.levels - 1) * d).mean()))
        iv = res.interval(0.1)
        cov = ((Y[:, N:] >= iv[:, :, 0]) & (Y[:, N:] <= iv[:, :, 1])).mean(0)
        tail.append(np.sort(cov)[:3].mean())
    cal, pin = np.mean(cal, 0), np.mean(pin, 0)
    assert cal[0] < 0.6 * cal[1] and pin[0] < pin[1] and np.mean(tail) > 0.8
    live = PanelPredictiveSystem(bandwidth=np.inf).start(n_cal=150, n_targets=30)
    q, b = live.predict(Y[0, :N] - P[0, :N])
    live.update(q, b, Y[0, N:] - P[0, N:], observed=np.ones(30, bool))
    assert q.shape == (30, 19)
