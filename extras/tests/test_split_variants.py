import math

import numpy as np
import pytest

import pyconformal as pc


def hetero(rng, n):
    x = rng.uniform(0, 1, n)
    sd = 0.2 + 2.0 * x
    return x, 3 * x + sd * rng.normal(size=n), sd


def cov(iv, y):
    return ((iv[:, 0] <= y) & (y <= iv[:, 1])).mean()


def test_normalized_covers_everywhere_and_is_narrower():
    rng = np.random.default_rng(0)
    plain_c, norm_c, lowx_plain, lowx_norm, w_plain, w_norm = [], [], [], [], [], []
    for _ in range(200):
        xc, yc, sc = hetero(rng, 300)
        xt, yt, st = hetero(rng, 500)
        a = pc.SplitConformalRegressor().calibrate(y_pred=3 * xc, y=yc).predict_interval(y_pred=3 * xt)
        b = pc.SplitConformalRegressor().calibrate(y_pred=3 * xc, y=yc, sigma=sc).predict_interval(y_pred=3 * xt, sigma=st)
        plain_c.append(cov(a, yt)); norm_c.append(cov(b, yt))
        hard = xt > 0.8
        lowx_plain.append(cov(a[hard], yt[hard])); lowx_norm.append(cov(b[hard], yt[hard]))
        w_plain.append(np.mean(a[:, 1] - a[:, 0])); w_norm.append(np.mean(b[:, 1] - b[:, 0]))
    for v in (plain_c, norm_c):
        assert np.mean(v) >= 0.9 - 3 * np.std(v) / math.sqrt(200)
    assert np.mean(lowx_plain) < 0.85 < np.mean(lowx_norm)  # plain intervals fail where noise is high
    assert np.mean(w_norm) < np.mean(w_plain)
    m = pc.SplitConformalRegressor().calibrate(y_pred=[0.0, 0.0, 0.0], y=[1.0, -2.0, 3.0], sigma=[1.0, 2.0, 1.0])
    np.testing.assert_array_equal(m.calibration_scores_, [1.0, 1.0, 3.0])
    np.testing.assert_allclose(m.predict_interval(y_pred=[10.0], sigma=[2.0], alpha=0.5), [[8.0, 12.0]])


def test_mondrian_covers_within_each_group():
    rng = np.random.default_rng(1)
    res = {0: [], 1: []}
    plain = {0: [], 1: []}
    for _ in range(200):
        g = rng.integers(0, 2, 600); y = rng.normal(size=600) * np.where(g == 1, 4.0, 1.0)
        gt = rng.integers(0, 2, 600); yt = rng.normal(size=600) * np.where(gt == 1, 4.0, 1.0)
        m = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(600), y=y, groups=g)
        iv = m.predict_interval(y_pred=np.zeros(600), groups=gt)
        iv0 = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(600), y=y).predict_interval(y_pred=np.zeros(600))
        for k in (0, 1):
            res[k].append(cov(iv[gt == k], yt[gt == k])); plain[k].append(cov(iv0[gt == k], yt[gt == k]))
    for k in (0, 1):
        assert np.mean(res[k]) >= 0.9 - 3 * np.std(res[k]) / math.sqrt(200)
        assert np.mean(res[k]) < 0.93
    assert np.mean(plain[1]) < 0.85 and np.mean(plain[0]) > 0.97  # without groups: one too low, one too high
    # each group equals a separate split conformal run
    sep = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros((g == 1).sum()), y=y[g == 1]).threshold()
    assert m.predict_interval(y_pred=[0.0], groups=[1])[0, 1] == sep
    small = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(12), y=np.arange(12.0), groups=["a"] * 10 + ["b"] * 2)
    out = small.predict_interval(y_pred=[0.0, 0.0], groups=["a", "b"])
    assert np.isfinite(out[0]).all() and np.isinf(out[1]).all()  # 2 points cannot give 90%
    with pytest.raises(ValueError, match="not seen"):
        small.predict_interval(y_pred=[0.0], groups=["c"])


def test_weighted_restores_coverage_under_covariate_shift():
    rng = np.random.default_rng(2)
    plain, weighted = [], []
    for _ in range(300):
        xc = rng.uniform(0, 1, 500); yc = xc * 3 * rng.normal(size=500)       # noise grows with x
        xt = rng.beta(4, 1, 500); yt = xt * 3 * rng.normal(size=500)           # test inputs lean to large x
        w = lambda x: 4 * x ** 3                                               # density ratio beta(4,1) / uniform
        zero = np.zeros(500)
        plain.append(cov(pc.SplitConformalRegressor().calibrate(y_pred=zero, y=yc).predict_interval(y_pred=zero), yt))
        m = pc.SplitConformalRegressor().calibrate(y_pred=zero, y=yc, weights=w(xc))
        weighted.append(cov(m.predict_interval(y_pred=zero, weights=w(xt)), yt))
    assert np.mean(plain) < 0.86
    assert np.mean(weighted) >= 0.9 - 3 * np.std(weighted) / math.sqrt(300)
    assert np.mean(weighted) < 0.93


def test_weighted_matches_definition_and_reduces_to_split():
    rng = np.random.default_rng(3)
    s = rng.exponential(size=40); w = rng.uniform(0.2, 3.0, 40); wt = rng.uniform(0.2, 3.0, 7)
    m = pc.SplitConformalRegressor(alpha=0.2).calibrate(y_pred=np.zeros(40), y=s, weights=w)
    got = m.predict_interval(y_pred=np.zeros(7), weights=wt)[:, 1]
    order = np.argsort(s)
    for i in range(7):
        p = np.append(w[order], wt[i]) / (w.sum() + wt[i])
        vals = np.append(s[order], np.inf)
        assert got[i] == vals[np.searchsorted(np.cumsum(p), 0.8 * (1 - 1e-12))]
    same = pc.SplitConformalRegressor(alpha=0.2).calibrate(y_pred=np.zeros(40), y=s, weights=np.ones(40))
    ref = pc.SplitConformalRegressor(alpha=0.2).calibrate(y_pred=np.zeros(40), y=s)
    np.testing.assert_array_equal(same.predict_interval(y_pred=np.zeros(3), weights=np.ones(3)), ref.predict_interval(y_pred=np.zeros(3)))
    heavy = m.predict_interval(y_pred=[0.0], weights=[1e9])  # a test point unlike any calibration point
    assert np.isinf(heavy).all()


def test_option_errors():
    m = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(30), y=np.arange(30.0), sigma=np.ones(30))
    with pytest.raises(ValueError, match="sigma"):
        m.predict_interval(y_pred=[0.0])
    with pytest.raises(ValueError, match="sigma"):
        pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(3), y=np.zeros(3)).predict_interval(y_pred=[0.0], sigma=[1.0])
    with pytest.raises(ValueError):
        pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(3), y=np.zeros(3), sigma=[1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(3), y=np.zeros(3), groups=[1, 1, 2], weights=np.ones(3))
    g = pc.SplitConformalRegressor().calibrate(y_pred=np.zeros(30), y=np.arange(30.0), groups=np.zeros(30))
    with pytest.raises(ValueError, match="groups"):
        g.predict_interval(y_pred=[0.0])
    with pytest.raises(RuntimeError):
        g.threshold()
    assert g.predict_interval(y_pred=[0.0, 1.0], groups=[0, 0], alpha=[0.5, 0.1]).shape == (2, 2, 2)
