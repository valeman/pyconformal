import math

import numpy as np
import pytest

import pyconformal as pc
from pyconformal.mstl import mstl, stl


def series(n=600):
    t = np.arange(n)
    return 5 + 0.01 * t + 2 * np.sin(2 * np.pi * t / 12) + np.sin(2 * np.pi * t / 84) + 0.5 * np.sin(t * t * 0.37)


AT = [0, 1, 49, 298, 299, 598, 599]


def test_matches_r_forecast_mstl():
    # values from forecast::mstl 9.0.2 on the same series, periods 12 and 84
    m = mstl(series(), [12, 84])
    np.testing.assert_allclose(m.seasonal[12][AT], [-0.040162875914324082, 0.878474704301864806, 0.879785590007247209, -1.708687429324262652, -1.104391186523469681, -1.748647688628566010, -1.106156622238438914], rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.seasonal[84][AT], [0.033100459280417570, 0.043265978412136259, -0.540990778376013393, -0.204752279934949044, -0.313011009079659563, 0.833840531383749806, 0.744638021703629582], rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.trend[AT], [5.0595701107948505, 5.0694919236837288, 5.5389957005184858, 7.9742664466863831, 7.9837806016934447, 10.9515109747104304, 10.9609121723332290], rtol=0, atol=1e-12)


def test_matches_r_stl():
    # values from R's stl(ts(y, frequency = 12), s.window = 7)
    s = stl(series(), 12, seasonal=7)
    np.testing.assert_allclose(s.seasonal[AT], [-0.051914954527887636, 0.835898046489534141, 0.905688291720797567, -1.795828143049166270, -1.186561383079729026, -1.631048653885149857, -1.209414243823804025], rtol=0, atol=1e-12)
    np.testing.assert_allclose(s.trend[AT], [5.1238846916697751, 5.1798595058836669, 5.1230365353179348, 7.7155672024277377, 7.6650627431984555, 11.5637640278771556, 11.6502560934608113], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "period,ns,deg,jump,n",
    [(7, 7, 0, False, 300), (7, 13, 1, True, 301), (12, 35, 0, True, 97), (24, 7, 1, False, 1000), (84, 15, 0, True, 600), (5, 101, 1, True, 60)],
)
def test_matches_statsmodels_stl(period, ns, deg, jump, n):
    sm = pytest.importorskip("statsmodels.tsa.seasonal")
    y = series(1000)[:n]
    odd = lambda v: int(v) + 1 - int(v) % 2
    nt, nl = odd(math.ceil(1.5 * period / (1 - 1.5 / ns))), odd(period + 1)
    j = (lambda w: math.ceil(w / 10)) if jump else (lambda w: 1)
    ref = sm.STL(y, period=period, seasonal=ns, trend=nt, low_pass=nl, seasonal_deg=deg, trend_deg=1, low_pass_deg=1,
                 seasonal_jump=j(ns), trend_jump=j(nt), low_pass_jump=j(nl)).fit(inner_iter=2, outer_iter=0)
    got = stl(y, period, ns, low_pass=nl, seasonal_deg=deg, seasonal_jump=j(ns), trend_jump=j(nt), low_pass_jump=j(nl))
    np.testing.assert_allclose(got.seasonal, ref.seasonal, rtol=0, atol=1e-10)
    np.testing.assert_allclose(got.trend, ref.trend, rtol=0, atol=1e-10)


def test_recovers_known_components():
    rng = np.random.default_rng(0)
    t = np.arange(24 * 7 * 12)
    day, week = 3 * np.sin(2 * np.pi * t / 24), 2 * np.cos(2 * np.pi * t / 168)
    y = 10 + 0.003 * t + day + week + 0.3 * rng.normal(size=t.size)
    m = mstl(y, [168, 24])  # any order
    assert list(m.seasonal) == [24, 168]
    np.testing.assert_allclose(m.observed, m.trend + m.seasonal_total + m.remainder, atol=1e-12)
    assert np.corrcoef(m.seasonal[24], day)[0, 1] > 0.99
    assert np.corrcoef(m.seasonal[168], week)[0, 1] > 0.98
    assert np.abs(m.trend - (10 + 0.003 * t)).mean() < 0.15
    assert m.remainder.std() < 0.4
    full = mstl(y, [24, 168], jumps=False)
    assert np.abs(full.seasonal[24] - m.seasonal[24]).max() < 0.1  # jumps change little


def test_periods_that_do_not_fit_are_dropped_and_errors():
    y = series(100)
    m = mstl(y, [12, 84])  # 84 does not fit twice in 100 points
    assert list(m.seasonal) == [12]
    np.testing.assert_allclose(m.seasonal[12], stl(y, 12, seasonal=11).seasonal, atol=1e-12)  # one period: plain STL
    with pytest.raises(ValueError):
        mstl(y, [84])
    with pytest.raises(ValueError):
        mstl(y, [1, 12])
    with pytest.raises(ValueError):
        stl(y, 12, seasonal=8)
    with pytest.raises(ValueError):
        stl(y[:20], 12)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        mstl(bad, [12])
    assert pc.mstl is mstl and pc.stl is stl
