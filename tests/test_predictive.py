import math
import warnings

import numpy as np
import pytest

from pyconformal import ConformalPredictiveSystem

pytestmark = pytest.mark.filterwarnings("error")


def make(rng, n, m, groups=False):
    pc, pt = rng.normal(size=n), rng.normal(size=m)
    sc, st = rng.uniform(0.5, 3, n), rng.uniform(0.5, 3, m)
    yc, yt = pc + sc * rng.standard_t(4, size=n), pt + st * rng.standard_t(4, size=m)
    gc = rng.integers(0, 3, n) if groups else None
    gt = rng.integers(0, 3, m) if groups else None
    return yc, pc, sc, gc, yt, pt, st, gt


def test_definitions_by_brute_force():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 40):
        yc, pc, sc, _, yt, pt, st, _ = make(rng, n, 30)
        yc = np.round(yc, 1); pc = np.round(pc, 1)  # ties
        cps = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc, sigma=sc)
        a = np.sort((yc - pc) / sc)
        C = cps.predict_distribution(y_pred=pt, sigma=st)
        assert np.allclose(C, pt[:, None] + st[:, None] * a[None, :])
        for q in (0.0, 0.03, 0.07, 0.25, 0.5, 0.93, 1.0):
            lo = cps.predict_quantile(y_pred=pt, q=q, sigma=st, side="lower")
            hi = cps.predict_quantile(y_pred=pt, q=q, sigma=st, side="higher")
            kl, kh = math.floor(round(q * (n + 1), 9)), math.ceil(round(q * (n + 1), 9))
            ext = np.concatenate([np.full((30, 1), -np.inf), C, np.full((30, 1), np.inf)], axis=1)
            assert np.array_equal(lo, ext[:, kl]) and np.array_equal(hi, ext[:, kh])
        # distribution function at support points and between them
        z = np.concatenate([a, a + 1e-3, [a.min() - 1]])
        for tau in (0.0, 0.3, 1.0):
            got = cps.predict_cdf(y_pred=np.zeros(z.size), y=z, sigma=np.ones(z.size), tau=tau)
            want = [(np.sum(a < v) + tau * (np.sum(a == v) + 1)) / (n + 1) for v in z]
            assert np.allclose(got, want, atol=1e-15)


def test_exact_rank_in_floating_point_traps():
    # 0.07 * 100 is 7.000000000000001 in floating point; the rank must be 7.
    y = np.arange(1.0, 100.0)
    cps = ConformalPredictiveSystem().calibrate(y=y, y_pred=np.zeros(99))
    assert cps.predict_quantile(y_pred=[0.0], q=0.07, side="higher")[0] == 7.0
    assert cps.predict_quantile(y_pred=[0.0], q=0.07, side="lower")[0] == 7.0
    iv = ConformalPredictiveSystem().calibrate(y=np.arange(1.0, 40.0), y_pred=np.zeros(39)).predict_interval(y_pred=[0.0], alpha=0.05)
    assert iv.tolist() == [[1.0, 39.0]]
    small = ConformalPredictiveSystem().calibrate(y=np.arange(1.0, 39.0), y_pred=np.zeros(38)).predict_interval(y_pred=[0.0], alpha=0.05)
    assert small[0, 0] == -np.inf and small[0, 1] == np.inf  # 38 points cannot give 95%


@pytest.mark.parametrize("normalized", [False, True])
def test_interval_coverage(normalized):
    rng = np.random.default_rng(1)
    n, alpha, covs = 99, 0.1, []
    for _ in range(300):
        yc, pc, sc, _, yt, pt, st, _ = make(rng, n, 400)
        if normalized:
            iv = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc, sigma=sc).predict_interval(y_pred=pt, alpha=alpha, sigma=st)
        else:
            iv = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc).predict_interval(y_pred=pt, alpha=alpha)
        covs.append(np.mean((iv[:, 0] <= yt) & (yt <= iv[:, 1])))
    # ranks 5 and 95 of 99: the outcome is equally likely to fall in any of
    # the 100 gaps, and 90 of them are inside, so coverage is exactly 0.90
    assert np.mean(covs) == pytest.approx(0.90, abs=0.005)


def test_probability_calibration():
    rng = np.random.default_rng(2)
    pit = []
    for _ in range(3000):  # few test points per calibration set: nearly independent
        yc, pc, sc, _, yt, pt, st, _ = make(rng, 50, 5)
        cps = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc, sigma=sc)
        pit.append(cps.predict_cdf(y_pred=pt, y=yt, sigma=st, tau=rng.uniform(size=5)))
    pit = np.sort(np.concatenate(pit))
    ks = np.max(np.abs(pit - (np.arange(pit.size) + 0.5) / pit.size))
    assert ks < 0.016  # uniform on [0, 1]
    # default tau gives values strictly inside (0, 1) and monotone in y
    cps = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc)
    grid = np.linspace(-20, 20, 500)
    F = cps.predict_cdf(y_pred=np.zeros(500), y=grid)
    assert np.all(np.diff(F) >= 0) and 0 < F[0] < F[-1] < 1


def test_mondrian_calibrates_each_group():
    rng = np.random.default_rng(3)
    cov = {0: [], 1: []}
    for _ in range(200):
        g_c, g_t = rng.integers(0, 2, 200), rng.integers(0, 2, 400)
        y_c = rng.normal(size=200) * np.where(g_c == 1, 5.0, 1.0)
        y_t = rng.normal(size=400) * np.where(g_t == 1, 5.0, 1.0)
        cps = ConformalPredictiveSystem().calibrate(y=y_c, y_pred=np.zeros(200), groups=g_c)
        iv = cps.predict_interval(y_pred=np.zeros(400), alpha=0.2, groups=g_t)
        hit = (iv[:, 0] <= y_t) & (y_t <= iv[:, 1])
        for g in (0, 1):
            cov[g].append(hit[g_t == g].mean())
    assert np.mean(cov[0]) >= 0.795 and np.mean(cov[1]) >= 0.795
    with pytest.raises(ValueError):
        cps.predict_interval(y_pred=[0.0], groups=[7])
    with pytest.raises(NotImplementedError):
        cps.predict_distribution(y_pred=[0.0])


@pytest.mark.parametrize("groups", [False, True])
def test_crps_matches_definition(groups):
    rng = np.random.default_rng(4)
    yc, pc, sc, gc, yt, pt, st, gt = make(rng, 60, 80, groups)
    yc = np.round(yc, 1)
    cps = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc, sigma=sc, groups=gc)
    got = cps.crps(y_pred=pt, y=yt, sigma=st, groups=gt)
    for i in range(80):
        sel = slice(None) if gc is None else gc == gt[i]
        C = pt[i] + st[i] * np.sort((yc[sel] - pc[sel]) / sc[sel])
        want = np.mean(np.abs(C - yt[i])) - 0.5 * np.mean(np.abs(C[:, None] - C[None, :]))
        assert got[i] == pytest.approx(want, rel=1e-10, abs=1e-12)
    assert np.all(got >= 0)


def test_agrees_with_crepes_where_crepes_is_right():
    crepes = pytest.importorskip("crepes")
    rng = np.random.default_rng(5)
    yc, pc, sc, gc, yt, pt, st, gt = make(rng, 57, 300, True)
    lp, hp = [2.5, 10, 33, 50], [50, 67, 90, 97.5]
    for use_sigma, use_groups in ((0, 0), (1, 0), (0, 1), (1, 1)):
        kw_c = dict(sigmas=sc if use_sigma else None, bins=gc if use_groups else None)
        kw_t = dict(sigmas=st if use_sigma else None, bins=gt if use_groups else None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = crepes.ConformalPredictiveSystem().fit(yc - pc, **kw_c)
            r = ref.predict(pt, lower_percentiles=lp, higher_percentiles=hp, **kw_t)
            pv = ref.predict(pt, y=yt, smoothing=False, **kw_t)
        ours = ConformalPredictiveSystem().calibrate(y=yc, y_pred=pc, sigma=sc if use_sigma else None, groups=gc if use_groups else None)
        s, g = (st if use_sigma else None), (gt if use_groups else None)
        o = np.column_stack([
            ours.predict_quantile(y_pred=pt, q=np.array(lp) / 100, sigma=s, groups=g, side="lower"),
            ours.predict_quantile(y_pred=pt, q=np.array(hp) / 100, sigma=s, groups=g, side="higher"),
        ])
        assert np.allclose(o, r, atol=1e-12)
        assert np.allclose(ours.predict_cdf(y_pred=pt, y=yt, sigma=s, groups=g, tau=1.0), np.ravel(pv), atol=1e-12)
        if not use_sigma and not use_groups:  # crepes' CRPS is only right without sigmas
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = ref.evaluate(pt, yt, metrics=["CRPS"])["CRPS"]
            assert ours.crps(y_pred=pt, y=yt).mean() == pytest.approx(c, rel=1e-9)


def test_validation():
    cps = ConformalPredictiveSystem()
    with pytest.raises(RuntimeError):
        cps.predict_interval(y_pred=[0.0])
    with pytest.raises(ValueError):
        cps.calibrate(y=[], y_pred=[])
    with pytest.raises(ValueError):
        cps.calibrate(y=[1.0, 2.0], y_pred=[1.0])
    with pytest.raises(ValueError):
        cps.calibrate(y=[1.0, 2.0], y_pred=[0.0, 0.0], sigma=[1.0, 0.0])
    cps.calibrate(y=[1.0, 2.0], y_pred=[0.0, 0.0], sigma=[1.0, 2.0])
    with pytest.raises(ValueError):
        cps.predict_interval(y_pred=[0.0])  # sigma required
    with pytest.raises(ValueError):
        cps.predict_quantile(y_pred=[0.0], q=1.5, sigma=[1.0])
    with pytest.raises(ValueError):
        cps.predict_cdf(y_pred=[0.0], y=0.0, sigma=[1.0], tau=2.0)
