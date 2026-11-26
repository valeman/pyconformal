"""Weighted conformal predictive systems (Jonkers et al., 2024)."""
import numpy as np
import pytest

from pyconformal import ConformalPredictiveSystem as CPS


def _brute_cdf(a, w, wt, z, tau):
    return (w[a < z].sum() + tau * (w[a == z].sum() + wt)) / (w.sum() + wt)


def _brute_quantile(a, w, wt, q, side):
    order = np.argsort(a, kind="stable"); a, w = a[order], w[order]; tot = w.sum() + wt; tol = 1e-12 * tot
    if side == "higher":                                   # smallest value with weight at or below it >= q * tot
        if q * tot - tol <= 0: return -np.inf
        for v in a:
            if w[a <= v].sum() >= q * tot - tol: return v
        return np.inf
    if w.sum() + wt <= q * tot + tol: return np.inf        # q = 1
    best = -np.inf                                         # largest value with weight at or above it >= (1 - q) * tot
    for v in a:
        if w[a >= v].sum() >= (1 - q) * tot - tol: best = max(best, v)
    return best


def test_equal_weights_give_the_unweighted_system():
    g = np.random.default_rng(0); n, m = 137, 400
    yc, pc, pt, yt = g.normal(size=n), g.normal(size=n), g.normal(size=m), g.normal(size=m)
    u = CPS().calibrate(y_pred=pc, y=yc); w = CPS().calibrate(y_pred=pc, y=yc, weights=np.full(n, 2.5))
    qs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0] + list(g.uniform(size=20))
    for side in ("lower", "higher"):
        assert np.array_equal(u.predict_quantile(y_pred=pt, q=qs, side=side), w.predict_quantile(y_pred=pt, q=qs, side=side, weights=2.5))
    assert np.array_equal(u.predict_interval(y_pred=pt, alpha=[0.1, 0.2]), w.predict_interval(y_pred=pt, alpha=[0.1, 0.2], weights=2.5))
    tau = g.uniform(size=m)
    assert np.allclose(u.predict_cdf(y_pred=pt, y=yt, tau=tau), w.predict_cdf(y_pred=pt, y=yt, tau=tau, weights=2.5), rtol=0, atol=1e-14)
    assert np.allclose(u.crps(y_pred=pt, y=yt), w.crps(y_pred=pt, y=yt), rtol=1e-12, atol=1e-14)


def test_integer_weights_equal_repeated_calibration_points():
    g = np.random.default_rng(1); n, m = 60, 200
    yc, pc = np.round(g.normal(size=n), 1), np.zeros(n); k = g.integers(0, 4, size=n); k[0] = 1
    pt, yt = g.normal(size=m), np.round(g.normal(size=m), 1)
    w = CPS().calibrate(y_pred=pc, y=yc, weights=k); r = CPS().calibrate(y_pred=np.zeros(k.sum()), y=np.repeat(yc, k))
    qs = [0.03, 0.1, 0.5, 0.9, 0.97]
    for side in ("lower", "higher"):
        assert np.array_equal(w.predict_quantile(y_pred=pt, q=qs, side=side, weights=1), r.predict_quantile(y_pred=pt, q=qs, side=side))
    assert np.allclose(w.predict_cdf(y_pred=pt, y=yt, weights=1, tau=0.3), r.predict_cdf(y_pred=pt, y=yt, tau=0.3), atol=1e-14)
    assert np.allclose(w.crps(y_pred=pt, y=yt), r.crps(y_pred=pt, y=yt), atol=1e-12)


def test_against_the_definition_with_ties_sigma_and_test_weights():
    g = np.random.default_rng(2)
    for _ in range(40):
        n = int(g.integers(1, 25)); a = np.round(g.normal(size=n), 1); w = g.uniform(0, 3, size=n) * (g.uniform(size=n) > 0.2); w[0] = max(w[0], 0.1)
        sig = g.uniform(0.5, 2, size=n); c = CPS().calibrate(y_pred=np.zeros(n), y=a * sig, sigma=sig, weights=w)
        m = 12; pt = g.normal(size=m); st = g.uniform(0.5, 2, size=m); wt = g.uniform(0, 3, size=m); z = np.round(g.normal(size=m), 1); tau = g.uniform(size=m)
        a = (a * sig) / sig; yq = pt + st * z; z = (yq - pt) / st      # the scores and thresholds as the class computes them
        cdf = c.predict_cdf(y_pred=pt, y=yq, sigma=st, weights=wt, tau=tau)
        qs = [0.0, 0.05, 0.3, 0.5, 0.8, 0.95, 1.0]
        lo = c.predict_quantile(y_pred=pt, q=qs, sigma=st, weights=wt, side="lower"); hi = c.predict_quantile(y_pred=pt, q=qs, sigma=st, weights=wt, side="higher")
        for i in range(m):
            assert abs(cdf[i] - _brute_cdf(a, w, wt[i], z[i], tau[i])) < 1e-12
            for j, q in enumerate(qs):
                for got, side in ((lo[i, j], "lower"), (hi[i, j], "higher")):
                    want = pt[i] + st[i] * _brute_quantile(a, w, wt[i], q, side)
                    assert got == want or abs(got - want) < 1e-12, (n, q, side, got, want)
        # CRPS of the distribution with mass w_i / W on each point, by direct summation
        p = w / w.sum(); y0 = g.normal()
        direct = (p * np.abs(a - y0)).sum() - 0.5 * (p[:, None] * p[None, :] * np.abs(a[:, None] - a[None, :])).sum()
        assert abs(c.crps(y_pred=np.zeros(1), y=np.array([y0]), sigma=np.ones(1))[0] - direct) < 1e-12


def test_covariate_shift_likelihood_ratio_weights_restore_calibration():
    g = np.random.default_rng(3); n, m, R = 500, 400, 200
    lr = lambda x: np.exp(x - 0.5)                         # density of N(1, 1) over density of N(0, 1)
    cov_w, cov_u, pit = [], [], []
    for _ in range(R):
        xc, xt = g.normal(size=n), g.normal(1.0, 1.0, size=m)
        yc, yt = (1 + np.abs(xc)) * g.normal(size=n), (1 + np.abs(xt)) * g.normal(size=m)     # noisier where the test data live
        w = CPS().calibrate(y_pred=np.zeros(n), y=yc, weights=lr(xc)); u = CPS().calibrate(y_pred=np.zeros(n), y=yc)
        iw = w.predict_interval(y_pred=np.zeros(m), alpha=0.1, weights=lr(xt)); iu = u.predict_interval(y_pred=np.zeros(m), alpha=0.1)
        cov_w.append(((iw[:, 0] <= yt) & (yt <= iw[:, 1])).mean()); cov_u.append(((iu[:, 0] <= yt) & (yt <= iu[:, 1])).mean())
        pit.append(w.predict_cdf(y_pred=np.zeros(m), y=yt, weights=lr(xt), tau=g.uniform(size=m)))
    assert np.mean(cov_u) < 0.86                           # the unweighted system under-covers under this shift
    assert 0.895 < np.mean(cov_w) < 0.93                   # the weighted one is back at its level
    pit = np.concatenate(pit); grid = np.linspace(0.05, 0.95, 19)
    assert np.max(np.abs((pit[:, None] <= grid[None, :]).mean(0) - grid)) < 0.01      # uniform


def test_weighted_mondrian_and_validation():
    g = np.random.default_rng(4); n = 80; yc = g.normal(size=n); grp = g.integers(0, 2, size=n); w = g.uniform(0.1, 2, size=n)
    c = CPS().calibrate(y_pred=np.zeros(n), y=yc, groups=grp, weights=w)
    for k in (0, 1):
        one = CPS().calibrate(y_pred=np.zeros((grp == k).sum()), y=yc[grp == k], weights=w[grp == k])
        assert np.array_equal(c.predict_quantile(y_pred=np.zeros(3), q=[0.1, 0.9], groups=np.full(3, k), weights=1.0), one.predict_quantile(y_pred=np.zeros(3), q=[0.1, 0.9], weights=1.0))
    with pytest.raises(ValueError): c.predict_quantile(y_pred=np.zeros(3), groups=np.zeros(3, int))            # test weights missing
    with pytest.raises(ValueError): CPS().calibrate(y_pred=np.zeros(n), y=yc).predict_cdf(y_pred=np.zeros(2), y=0.0, weights=1.0)
    with pytest.raises(ValueError): CPS().calibrate(y_pred=np.zeros(n), y=yc, weights=-w)
    with pytest.raises(ValueError): CPS().calibrate(y_pred=np.zeros(n), y=yc, weights=np.zeros(n))
    with pytest.raises(ValueError): CPS().calibrate(y_pred=np.zeros(n), y=yc, weights=w[:5])
    z = CPS().calibrate(y_pred=np.zeros(3), y=np.array([1.0, 2.0, 3.0]), weights=[1, 1, 1])
    assert np.isinf(z.predict_quantile(y_pred=np.zeros(1), q=0.9, weights=100.0)[0])       # a heavy test point: no finite upper quantile
    assert z.predict_quantile(y_pred=np.zeros(1), q=0.9, weights=0.0)[0] == 3.0             # a weightless one: the plain weighted quantile
