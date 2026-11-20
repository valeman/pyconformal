import numpy as np
import pytest

from pyconformal import ConformalSeasonalPool

pytestmark = pytest.mark.filterwarnings("error")


def materialise(model, H):
    """Expand the vectorised plan into explicit per-horizon pools."""
    y, m, plan = model.history, model.seasonal_period, model._plan(H)
    out = []
    for h in range(H):
        P = plan["pool_size"][h]
        pool = y[plan["pool_last"][h] - m * np.arange(P)] if P > 0 else np.empty(0)
        w = np.exp(model._log_rho() * np.arange(P))
        a, lag, size, scale = (plan[k][h] for k in ("res_start", "res_lag", "res_size", "res_scale"))
        res = (y[a + lag : a + lag + size] - y[a : a + size]) * scale if size > 0 else np.array([0.0])
        out.append((pool, w / w.sum() if P else w, res, plan["mu"][h]))
    return out


CASES = [
    (T, m, H, cal, rmode, unit, lam)
    for T in (1, 2, 5, 9, 14, 23, 60, 200)
    for m in (1, 4, 7, 12, 24)
    for H in (1, 6, 30)
    for cal, rmode, unit, lam in (
        (0.5, "h_step", "step", 0.01),
        (0.5, "paper", "cycle", 0.01),
        (0.3, "h_step", "cycle", 0.0),
        (1.0, "h_step", "step", 0.2),
    )
]


def test_same_distribution_as_csp_forecaster():
    """Pools, recency weights, residual pools and point forecasts equal those
    of csp-forecaster's fast mode, horizon by horizon."""
    ref_mod = pytest.importorskip("csp_forecaster")
    rng = np.random.default_rng(0)
    for T, m, H, cal, rmode, unit, lam in CASES:
        y = rng.normal(size=T).cumsum() + 10 * np.sin(np.arange(T))
        kw = dict(cal_fraction=cal, residual_mode=rmode, decay_unit=unit, exp_lambda=lam)
        ours = ConformalSeasonalPool(**kw).fit(y, m)
        ref = ref_mod.ConformalSeasonalPool(mode="fast", **kw).fit(y, m)
        ref_pools = ref._residual_pools(H)
        assert ours._effective_pool_weight() == ref._effective_pool_weight()
        for h, (pool, w, res, mu) in enumerate(materialise(ours, H)):
            tag = (T, m, H, cal, rmode, unit, lam, h)
            assert mu == ref._seasonal_point_forecast(h), tag
            ref_res = ref._residuals() if ref_pools is None else ref_pools[h]
            assert np.array_equal(res, ref_res), tag
            idx = np.where(np.arange(T) % ours.seasonal_period == (T + h) % ours.seasonal_period)[0]
            assert np.array_equal(pool[::-1], y[idx]), tag
            if len(idx) and lam > 0:
                ages = (idx[-1] - idx) if unit == "step" else (idx // ours.seasonal_period)[-1] - idx // ours.seasonal_period
                rw = np.exp(-lam * ages.astype(float))
                assert np.allclose(w[::-1], rw / rw.sum(), rtol=1e-12, atol=0), tag


@pytest.mark.parametrize("lam,unit", [(0.01, "step"), (0.3, "cycle"), (0.0, "step"), (50.0, "step")])
def test_samples_follow_the_planned_distribution(lam, unit):
    rng = np.random.default_rng(1)
    y = rng.normal(size=300).cumsum()
    m, H, n = 12, 20, 400_000
    model = ConformalSeasonalPool(exp_lambda=lam, decay_unit=unit, random_state=0).fit(y, m)
    res = model.predict(H, n_samples=n)
    n_pool = int(n * model._effective_pool_weight())
    for h, (pool, w, r, mu) in enumerate(materialise(model, H)):
        if h not in (0, 7, 19):
            continue
        # seasonal-pool part: frequency of each value matches its weight
        got = res.samples[h, :n_pool]
        for v, wv in zip(pool[:6], w[:6]):
            f = np.mean(got == v)
            assert abs(f - wv) < 5 * np.sqrt(wv * (1 - wv) / n_pool) + 1e-9
        assert np.isin(got, pool).all()
        # conformal part: uniform over the residual pool
        conf = res.samples[h, n_pool:]
        assert np.isin(np.round(conf - mu, 9), np.round(r, 9)).all()
        assert abs(conf.mean() - (mu + r.mean())) < 5 * r.std() / np.sqrt(conf.size) + 1e-9


def test_statistically_equivalent_to_csp_forecaster():
    ref_mod = pytest.importorskip("csp_forecaster")
    rng = np.random.default_rng(2)
    for T, m, H in ((500, 24, 48), (120, 1, 12), (40, 7, 21)):
        y = rng.normal(size=T).cumsum() + 5 * np.sin(2 * np.pi * np.arange(T) / max(m, 2))
        ours = ConformalSeasonalPool(random_state=1).fit(y, m).predict(H, n_samples=200_000)
        ref = ref_mod.ConformalSeasonalPool(mode="fast", random_state=2).fit(y, m).predict(H, n_samples=200_000)
        # The distribution is discrete, so quantiles that fall in a gap between
        # atoms are ill-conditioned. Compare the distribution functions instead
        # (two-sample Kolmogorov-Smirnov distance per horizon).
        for h in range(H):
            a = np.sort(ours.samples[h])
            b = np.sort(ref.samples[h].astype(np.float64))
            grid = np.concatenate([a, b])
            # csp-forecaster stores float32; evaluate just above each atom
            grid = grid + 1e-4 * (np.abs(grid) + 1.0)
            d = np.abs(np.searchsorted(a, grid) - np.searchsorted(b, grid)).max() / a.size
            assert d < 0.008, (T, m, h, d)


def test_quantiles_intervals_and_options():
    rng = np.random.default_rng(3)
    y = rng.normal(size=200).cumsum()
    model = ConformalSeasonalPool(random_state=0).fit(y, 12)
    res = model.predict(24, alpha=0.1, quantile_levels=[0.1, 0.5, 0.9], n_samples=101)
    taus = [0.05, 0.1, 0.5, 0.9, 0.95]
    ref = np.quantile(res.samples, taus, axis=1)
    assert np.allclose(res.lower, ref[0]) and np.allclose(res.upper, ref[4])
    assert np.allclose(res.quantiles[0.5], ref[2])
    assert np.all(res.lower <= res.upper) and res.samples.shape == (24, 101)
    # same seed, same output
    again = ConformalSeasonalPool(random_state=0).fit(y, 12).predict(24, alpha=0.1, n_samples=101)
    assert np.array_equal(res.samples, again.samples)
    # orientation only moves the bounds outward
    o = ConformalSeasonalPool(random_state=0, orientation=True).fit(y, 12).predict(24, alpha=0.1, n_samples=101)
    assert np.array_equal(o.samples, res.samples)
    assert np.all(o.lower <= res.lower) and np.all(o.upper >= res.upper)
    # h_step intervals widen with horizon on a non-seasonal random walk
    rw = ConformalSeasonalPool(random_state=0).fit(y, 1).predict(30, n_samples=2000)
    width = rw.upper - rw.lower
    assert width[-5:].mean() > 2 * width[:5].mean()


def test_tiny_and_extreme_inputs():
    for y, m in (([3.0], 1), ([3.0], 12), ([1.0, 2.0], 7), ([1e20, -1e20, 1e20, 5.0], 2)):
        res = ConformalSeasonalPool(random_state=0).fit(y, m).predict(5, n_samples=50)
        assert np.all(np.isfinite(res.samples))
    assert np.all(ConformalSeasonalPool().fit([3.0], 1).predict(3).samples == 3.0)
    with pytest.raises(ValueError):
        ConformalSeasonalPool().fit([])
    with pytest.raises(ValueError):
        ConformalSeasonalPool().fit([1.0, np.nan])
    with pytest.raises(RuntimeError):
        ConformalSeasonalPool().predict(3)
    with pytest.raises(ValueError):
        ConformalSeasonalPool(residual_mode="x")
