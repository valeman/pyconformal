"""Conformal prediction for distribution-valued outcomes."""

import math
from fractions import Fraction

import numpy as np
import pyconformal as pc
import pytest

from pyconformal import ConformalDistributionRegressor as CDR
from pyconformal import cosine_cdf, cosine_coefficients, cosine_density


def test_coefficients_match_the_definition_and_input_forms():
    rng = np.random.default_rng(0)
    S = rng.beta(2, 5, size=(7, 300))
    A = cosine_coefficients(S, n_terms=6)
    want = np.array([[np.mean(math.sqrt(2) * np.cos(math.pi * k * s)) for k in range(1, 7)] for s in S])
    np.testing.assert_allclose(A, want, rtol=0, atol=1e-13)
    ragged = [S[0], S[1, :100], S[2, :250]]
    R = cosine_coefficients(ragged, n_terms=6)
    np.testing.assert_allclose(R[0], A[0], atol=1e-13)
    np.testing.assert_allclose(R[1], cosine_coefficients(S[1, :100], 6)[0], atol=1e-13)
    # another support is a rescaling
    np.testing.assert_allclose(cosine_coefficients(10 + 5 * S, 6, support=(10, 15)), A, atol=1e-12)
    with pytest.raises(ValueError):
        cosine_coefficients(S + 1.0, 6)


def test_series_recovers_the_distribution():
    x = np.random.default_rng(1).beta(3, 4, size=200_000)
    a = cosine_coefficients(x, n_terms=25)
    grid = np.linspace(0.05, 0.95, 19)
    emp = (x[None, :] <= grid[:, None]).mean(axis=1)
    np.testing.assert_allclose(cosine_cdf(a, grid)[0], emp, atol=5e-3)
    assert cosine_cdf(a, [0.0, 1.0])[0] == pytest.approx([0.0, 1.0], abs=1e-12)
    d = cosine_density(a, np.linspace(0, 1, 2001))[0]
    assert np.sum((d[1:] + d[:-1]) / 2) / 2000 == pytest.approx(1.0, abs=1e-6)


def test_radius_is_the_exact_conformal_rank_and_infinite_when_too_few():
    rng = np.random.default_rng(2)
    for n in (5, 9, 19, 50):
        Y, P = rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
        s = np.linalg.norm(Y - P, axis=1)
        for a in (0.1, 0.05, 0.3):
            k = math.ceil((n + 1) * (1 - Fraction(repr(a))))
            want = np.inf if k > n else np.sort(s)[k - 1]
            assert CDR(alpha=a).calibrate(y_pred=P, y=Y).radius() == pytest.approx(want, rel=1e-12)
    m = CDR(alpha=0.1).calibrate(y_pred=np.zeros((5, 2)), y=np.ones((5, 2)))
    c, r = m.predict_ball(y_pred=np.zeros((3, 2)))
    assert np.isinf(r).all() and m.contains(np.full((3, 2), 1e9), y_pred=np.zeros((3, 2))).all()


def test_marginal_coverage_over_many_calibration_draws():
    rng = np.random.default_rng(3)
    hits = []
    for _ in range(400):
        Y = rng.normal(size=(140, 5)) * rng.gamma(2, 1, size=(140, 1))
        m = CDR(alpha=0.1).calibrate(y_pred=np.zeros((40, 5)), y=Y[:40])
        hits.append(m.contains(Y[40:], y_pred=np.zeros((100, 5))).mean())
    # exact value for n = 40: ceil(41 * 0.9) / 41 = 37 / 41
    assert np.mean(hits) == pytest.approx(37 / 41, abs=0.01)


def test_with_a_model_on_distribution_data():
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(4)
    a, b = rng.uniform(1, 6, 1500), rng.uniform(1, 6, 1500)
    Xs = rng.beta(a[:, None], b[:, None], size=(1500, 300))
    Ys = np.clip(Xs**1.5 + 0.03 * rng.normal(size=(1500, 1)), 0, 1)  # a shift shared within each distribution
    A_in, A_out = cosine_coefficients(Xs, 8), cosine_coefficients(Ys, 8)
    cd = CDR(Ridge(1e-3), alpha=0.1).fit(A_in[:500], A_out[:500]).calibrate(A_in[500:1000], A_out[500:1000])
    assert 0.86 < cd.contains(A_out[1000:], A_in[1000:]).mean() < 0.94
    # the band contains the truncated-series CDF whenever the ball contains the coefficients
    grid = np.linspace(0, 1, 41)
    lo, hi = cd.predict_cdf_band(grid, A_in[1000:])
    F = np.clip(cosine_cdf(A_out[1000:], grid), 0.0, 1.0)
    inside_ball = cd.contains(A_out[1000:], A_in[1000:])
    in_band = np.all((F >= lo - 1e-12) & (F <= hi + 1e-12), axis=1)
    assert np.all(in_band[inside_ball]) and in_band.mean() >= inside_ball.mean()
    assert np.all(lo <= hi) and lo.min() >= 0 and hi.max() <= 1


def test_band_bounds_are_attained_inside_the_ball():
    P = np.array([[0.1, -0.2, 0.05]])
    m = CDR(alpha=0.5).calibrate(y_pred=np.zeros((9, 3)), y=np.eye(3)[[0] * 9] * 0.02)
    r = m.radius()
    grid = np.array([0.3])
    lo, hi = m.predict_cdf_band(grid, y_pred=P)
    from pyconformal.distribution import _cdf_weights

    w = _cdf_weights(grid, 3, (0, 1))[1][0]
    top = P + r * w / np.linalg.norm(w)  # the vector in the ball that maximises F(0.3)
    assert cosine_cdf(top, grid)[0, 0] == pytest.approx(hi[0, 0], abs=1e-12)
    assert np.linalg.norm(top - P) == pytest.approx(r)


def test_localised_radius_matches_a_direct_computation_and_adapts():
    rng = np.random.default_rng(5)
    n, K = 600, 80
    F = rng.uniform(-1, 1, size=(n + 200, 2))
    noise = 0.1 + 2.0 * (F[:, 0] > 0)  # errors are larger on one side
    Y = rng.normal(size=(n + 200, 3)) * noise[:, None]
    P = np.zeros_like(Y)
    m = CDR(alpha=0.1, k_neighbors=K).calibrate(y_pred=P[:n], y=Y[:n], features=F[:n])
    r = m.radius(features=F[n:])
    s = np.linalg.norm(Y[:n], axis=1)
    k = math.ceil((K + 1) * Fraction(9, 10))
    for i in range(0, 200, 7):
        near = np.argsort(np.sum((F[:n] - F[n + i]) ** 2, axis=1), kind="stable")[:K]
        assert r[i] == pytest.approx(np.sort(s[near])[k - 1], rel=1e-12)
    assert r[F[n:, 0] > 0.3].mean() > 5 * r[F[n:, 0] < -0.3].mean()
    cov = m.contains(Y[n:], y_pred=P[n:], features=F[n:])
    assert cov[F[n:, 0] > 0.3].mean() > 0.8 and cov[F[n:, 0] < -0.3].mean() > 0.8
    # too few neighbours for the level: infinite, never a clamped rank
    assert np.isinf(CDR(alpha=0.1, k_neighbors=5).calibrate(y_pred=P[:n], y=Y[:n], features=F[:n]).radius(features=F[n:])).all()


def test_errors():
    with pytest.raises(RuntimeError):
        CDR().predict_ball(y_pred=np.zeros((1, 2)))
    with pytest.raises(ValueError):
        CDR().calibrate(y_pred=np.zeros((5, 2)), y=np.zeros((5, 3)))
    with pytest.raises(TypeError):
        CDR(k_neighbors=3).calibrate(y_pred=np.zeros((5, 2)), y=np.zeros((5, 2)))
    with pytest.raises(ValueError):
        CDR().calibrate(y_pred=np.zeros((5, 2)), y=np.zeros((5, 2))).predict_ball(y_pred=np.zeros((1, 3)))


def test_weights_give_an_ellipsoid_and_the_volume_formula():
    rng = np.random.default_rng(6)
    Y, P = rng.normal(size=(99, 4)), np.zeros((99, 4))
    w = np.array([1.0, 4.0, 9.0, 16.0])
    m = CDR(alpha=0.1, weights=w).calibrate(y_pred=P, y=Y)
    s = np.sqrt(((Y - P) ** 2 * w).sum(axis=1))
    assert m.radius() == pytest.approx(np.sort(s)[89], rel=1e-12)
    assert m.contains(Y, y_pred=P).sum() == 90
    want = 2 * math.log(math.pi) + 4 * math.log(m.radius()) - math.lgamma(3.0) - 0.5 * np.log(w).sum()
    assert m.log_volume() == pytest.approx(want)
    # two outputs, no weights: a disc of area pi r^2
    d = CDR(alpha=0.5).calibrate(y_pred=np.zeros((9, 2)), y=np.ones((9, 2)))
    assert math.exp(d.log_volume()) == pytest.approx(math.pi * d.radius() ** 2)
    # the band with weights is still attained inside the ellipsoid
    from pyconformal.distribution import _cdf_weights

    c = _cdf_weights([0.4], 4, (0, 1))[1][0]
    m = CDR(alpha=0.1, weights=w).calibrate(y_pred=P, y=0.01 * Y)  # small enough not to be cut at 1
    lo, hi = m.predict_cdf_band([0.4], y_pred=P[:1])
    top = P[:1] + m.radius() * (c / w) / np.sqrt(np.sum(c * c / w))
    assert np.sqrt(((top - P[:1]) ** 2 * w).sum()) == pytest.approx(m.radius())
    assert cosine_cdf(top, [0.4])[0, 0] == pytest.approx(hi[0, 0], abs=1e-12)


def test_compiled_and_plain_coefficients_agree():
    from pyconformal import distribution as D

    S = np.random.default_rng(7).random((50, 777))
    a = np.empty((50, 12)); b = np.empty((50, 12))
    assert D._coef_py(S, a) == 0
    if D._HAVE_NUMBA:
        assert D._coef_nb(S, b) == 0
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
    bad = S.copy(); bad[3, 5] = np.nan
    assert D._coef_py(bad, a) == 1
    bad[3, 5] = 1.5
    assert D._coef_py(bad, a) == 2


def test_several_levels_at_once_equal_one_at_a_time():
    rng = np.random.default_rng(8)
    Y, F = rng.normal(size=(400, 3)), rng.normal(size=(400, 2))
    levels = [0.3, 0.1, 0.05, 0.01]
    g = CDR().calibrate(y_pred=np.zeros((300, 3)), y=Y[:300])
    np.testing.assert_array_equal(g.radius(alpha=levels), [g.radius(alpha=a) for a in levels])
    loc = CDR(k_neighbors=50).calibrate(y_pred=np.zeros((300, 3)), y=Y[:300], features=F[:300])
    R = loc.radius(alpha=levels, features=F[300:])
    assert R.shape == (100, 4)
    for j, a in enumerate(levels):
        np.testing.assert_array_equal(R[:, j], loc.radius(alpha=a, features=F[300:]))
    assert np.isinf(R[:, 3]).all()  # 50 neighbours cannot give 99%


def test_compiled_and_plain_localised_radii_agree():
    from pyconformal import distribution as D

    if not D._HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(9)
    Y = rng.normal(size=(4600, 3))
    F = rng.normal(size=(4600, 4)) * np.array([1.0, 0.5, 0.05, 0.01])  # two main directions
    loc = CDR(k_neighbors=120).calibrate(y_pred=np.zeros((4000, 3)), y=Y[:4000], features=F[:4000])
    assert loc._use_tree(600)
    fast = loc.radius(alpha=[0.2, 0.1, 0.001], features=F[4000:])  # tree search
    S = loc._neighbours(F[4000:])
    S.sort(axis=1)
    np.testing.assert_allclose(fast[:, 0], S[:, 96], rtol=1e-12)   # ceil(121 * 0.8) = 97
    np.testing.assert_allclose(fast[:, 1], S[:, 108], rtol=1e-12)  # ceil(121 * 0.9) = 109
    assert np.isinf(fast[:, 2]).all()


def test_tree_search_with_many_identical_rows():
    from pyconformal import distribution as D

    if not D._HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(10)
    F = np.repeat(rng.normal(size=(40, 2)) * np.array([1.0, 0.01]), 100, axis=0)  # 40 points, 100 copies each
    s = np.abs(rng.normal(size=4000))
    m = CDR(k_neighbors=100)
    m._features, m.calibration_scores_, m.n_outputs_ = F, s, 1
    assert m._use_tree(600)
    r = m.radius(features=F[::100][np.arange(600) % 40])
    # the 100 nearest rows of each query are exactly its own 100 copies
    want = np.array([np.sort(s[i * 100:(i + 1) * 100])[90] for i in range(40)])  # ceil(101 * 0.9) = 91
    np.testing.assert_allclose(r, want[np.arange(600) % 40], rtol=1e-12)
