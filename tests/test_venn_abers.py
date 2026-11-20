import warnings
from fractions import Fraction

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

from pyconformal import VennAbers as VennAbersCalibrator
from pyconformal.venn_abers import _IVAP as VennAbers
from pyconformal.venn_abers import _MulticlassIVAP as MulticlassVennAbers
from pyconformal._va_kernel import HAVE_NUMBA

# Every test runs with warnings as errors: Venn-Abers must be numerically clean.
pytestmark = pytest.mark.filterwarnings("error")


def brute_force(scores, labels, s, label):
    """Isotonic fit at a new point (s, label), by the max-min formula in
    exact rational arithmetic. Independent of the package's algorithm."""
    pts = sorted(list(zip(scores, labels)) + [(s, label)])
    xs = sorted(set(p[0] for p in pts))
    w = [sum(1 for p in pts if p[0] == x) for x in xs]
    t = [sum(p[1] for p in pts if p[0] == x) for x in xs]
    g = xs.index(s)
    best = None
    for a in range(g + 1):
        inner = min(
            Fraction(sum(t[a : b + 1]), sum(w[a : b + 1])) for b in range(g, len(xs))
        )
        best = inner if best is None else max(best, inner)
    return best


KERNELS = [False] + ([True] if HAVE_NUMBA else [])


@pytest.mark.parametrize("use_numba", KERNELS)
def test_matches_brute_force(use_numba):
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(1, 25))
        # few distinct values -> many ties
        scores = rng.integers(0, int(rng.integers(1, 12)) + 1, size=n) / 10.0
        labels = (rng.uniform(size=n) < rng.uniform()).astype(int)
        va = VennAbers().fit(scores, labels, use_numba=use_numba)
        test = np.unique(np.concatenate([scores, scores + 0.05, [-1.0, 5.0]]))
        got = va.predict_p0_p1(test)
        for s, (p0, p1) in zip(test, got):
            e0 = brute_force(scores.tolist(), labels.tolist(), float(s), 0)
            e1 = brute_force(scores.tolist(), labels.tolist(), float(s), 1)
            assert p0 == pytest.approx(float(e0), abs=1e-14), (trial, s)
            assert p1 == pytest.approx(float(e1), abs=1e-14), (trial, s)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_numba_and_python_kernels_identical():
    rng = np.random.default_rng(1)
    s = np.round(rng.uniform(size=5000), 3)
    y = rng.uniform(size=5000) < s**2
    a = VennAbers().fit(s, y, use_numba=True)
    b = VennAbers().fit(s, y, use_numba=False)
    assert np.array_equal(a.p0_, b.p0_) and np.array_equal(a.p1_, b.p1_)


def test_properties():
    rng = np.random.default_rng(2)
    s = rng.normal(size=3000)  # arbitrary real scores, not probabilities
    y = rng.uniform(size=3000) < 1 / (1 + np.exp(-2 * s))
    va = VennAbers().fit(s, y)
    grid = np.linspace(-5, 5, 2001)
    pp = va.predict_p0_p1(grid)
    assert np.all(pp >= 0) and np.all(pp <= 1)
    assert np.all(pp[:, 0] <= pp[:, 1])
    assert np.all(np.diff(pp, axis=0) >= 0)  # both monotone in the score
    for merge in ("log", "brier"):
        p = VennAbers(merge).fit(s, y).predict_proba(grid)
        assert np.allclose(p.sum(axis=1), 1)
        assert np.all(p[:, 1] >= pp[:, 0] - 1e-15) and np.all(p[:, 1] <= pp[:, 1] + 1e-15)


def test_degenerate_inputs():
    one = VennAbers().fit([0.3], [1])
    assert one.predict_p0_p1([0.1, 0.3, 0.9]).tolist() == [[0, 1], [0.5, 1], [0.5, 1]]
    all_pos = VennAbers().fit([0.1, 0.5, 0.9], [1, 1, 1])
    assert np.all(all_pos.predict_p0_p1([0.5])[:, 1] == 1)
    all_neg = VennAbers().fit([0.1, 0.5, 0.9], [0, 0, 0])
    assert np.all(all_neg.predict_p0_p1([0.5])[:, 0] == 0)
    same = VennAbers().fit([0.5] * 4, [0, 1, 1, 0])
    assert same.predict_p0_p1([0.5]).tolist() == [[0.4, 0.6]]
    # (n, 2) probability arrays are accepted
    P = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert np.array_equal(
        VennAbers().fit(P, [0, 1]).p1_, VennAbers().fit(P[:, 1], [0, 1]).p1_
    )


def test_input_validation():
    with pytest.raises(ValueError):
        VennAbers().fit([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError):
        VennAbers().fit([0.1, np.nan], [0, 1])
    with pytest.raises(ValueError):
        VennAbers().fit([], [])
    with pytest.raises(ValueError):
        VennAbers().fit([0.1, 0.2], [0])
    with pytest.raises(ValueError):
        VennAbers(merge="other")
    with pytest.raises(RuntimeError):
        VennAbers().predict_proba([0.5])
    with pytest.raises(ValueError):
        MulticlassVennAbers().fit(np.full((4, 3), 1 / 3), [0, 1, 0, 1])  # class 2 absent


def test_improves_calibration_of_overconfident_model():
    X, y = make_classification(n_samples=6000, n_features=20, n_informative=5, random_state=0)
    model = VennAbersCalibrator(GaussianNB()).fit(X[:2000], y[:2000])
    model.calibrate(X[2000:4000], y[2000:4000])
    raw = model.estimator.predict_proba(X[4000:])[:, 1]
    cal = model.predict_proba(X[4000:])[:, 1]
    yt = y[4000:]
    assert np.mean((cal - yt) ** 2) < np.mean((raw - yt) ** 2)
    pp = model.predict_p0_p1(X[4000:])
    assert pp.shape == (2000, 2)


@pytest.mark.parametrize("strategy", ["one_vs_one", "one_vs_rest"])
def test_multiclass(strategy):
    X, y = make_classification(
        n_samples=6000, n_features=20, n_informative=8, n_classes=4, random_state=1
    )
    labels = np.array(["a", "b", "c", "d"])[y]  # non-integer labels
    model = VennAbersCalibrator(GaussianNB(), strategy=strategy).fit(X[:2000], labels[:2000])
    model.calibrate(X[2000:4000], labels[2000:4000])
    P = model.predict_proba(X[4000:])
    assert P.shape == (2000, 4) and np.allclose(P.sum(axis=1), 1) and np.all(P >= 0)
    raw = model.estimator.predict_proba(X[4000:])
    onehot = labels[4000:, None] == model.classes_[None, :]
    assert np.mean((P - onehot) ** 2) < np.mean((raw - onehot) ** 2)
    assert set(model.predict(X[:5])) <= set(labels)
    with pytest.raises(RuntimeError):
        model.predict_p0_p1(X[:5])


def test_multiclass_handles_zero_probabilities():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=300)
    P = np.eye(3)[np.where(rng.uniform(size=300) < 0.8, y, rng.integers(0, 3, size=300))]
    out = MulticlassVennAbers().fit(P, y).predict_proba(P)  # hard 0/1 inputs
    assert np.all(np.isfinite(out)) and np.allclose(out.sum(axis=1), 1)


# ------------------------------------------------- agreement with venn-abers


def test_agrees_with_venn_abers_package():
    va_pkg = pytest.importorskip("venn_abers")
    rng = np.random.default_rng(4)
    for n in (50, 2000):
        s_cal = np.round(rng.uniform(size=n), 2 if n == 50 else 6)
        y_cal = (rng.uniform(size=n) < s_cal**1.5).astype(int)
        s_te = np.concatenate([rng.uniform(size=500), s_cal[:50]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = va_pkg.VennAbers()
            ref.fit(np.column_stack([1 - s_cal, s_cal]), y_cal)
            ref_p, ref_pp = ref.predict_proba(np.column_stack([1 - s_te, s_te]))
        ours = VennAbers().fit(s_cal, y_cal)
        assert np.allclose(ours.predict_p0_p1(s_te), ref_pp, atol=1e-12)
        assert np.allclose(ours.predict_proba(s_te), ref_p, atol=1e-12)


def test_multiclass_agrees_with_venn_abers_package():
    va_pkg = pytest.importorskip("venn_abers")
    rng = np.random.default_rng(5)
    K, n = 4, 3000
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K)) + 2.0 * np.eye(K)[y]
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    for strategy in ("one_vs_one", "one_vs_rest"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = va_pkg.VennAbersCalibrator().predict_proba(
                p_cal=P[:2000], y_cal=y[:2000], p_test=P[2000:], va_type=strategy
            )
        ours = MulticlassVennAbers(strategy).fit(P[:2000], y[:2000]).predict_proba(P[2000:])
        assert np.allclose(ours, ref, atol=1e-10), strategy


# ------------------------------------------------------ cross Venn-Abers


def _cva_data(rng, n):
    X = rng.normal(size=(n, 5))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] ** 2 - 0.5)))
    return X, (rng.uniform(size=n) < p).astype(int)


def test_cross_venn_abers_matches_reference_package():
    va = pytest.importorskip("venn_abers")
    from sklearn.naive_bayes import GaussianNB
    import pyconformal as pc

    rng = np.random.default_rng(0)
    X, y = _cva_data(rng, 600)
    Xt, _ = _cva_data(rng, 300)
    ours = pc.CrossVennAbers(GaussianNB(), cv=5, random_state=0).fit(X, y)
    # the reference, fed exactly our folds: per-fold calibration scores and per-fold test scores
    S = np.column_stack([m.predict_proba(Xt)[:, 1] for m in ours.estimators_])
    pairs = ours.predict_pairs(Xt)
    order = np.random.default_rng(0).permutation(600)
    folds = np.empty(600, dtype=int)
    folds[order] = np.arange(600) % 5
    for k, m in enumerate(ours.estimators_):
        cal = folds == k
        ref = va.VennAbers()
        with warnings.catch_warnings():  # the reference package warns about its own arithmetic
            warnings.simplefilter("ignore")
            ref.fit(m.predict_proba(X[cal]), y[cal])
            _, p01 = ref.predict_proba(np.column_stack([1 - S[:, k], S[:, k]]))
        np.testing.assert_allclose(pairs[:, k, :], p01, rtol=0, atol=1e-12)
    p0, p1 = pairs[:, :, 0], pairs[:, :, 1]
    gm = lambda a: np.exp(np.mean(np.log(a), axis=1))
    np.testing.assert_allclose(ours.predict_proba(Xt)[:, 1], gm(p1) / (gm(1 - p0) + gm(p1)), atol=1e-12)
    brier = pc.CrossVennAbers(GaussianNB(), cv=5, merge="brier", random_state=0).fit(X, y)
    np.testing.assert_allclose(brier.predict_proba(Xt)[:, 1], np.mean(p1 + p0**2 / 2 - p1**2 / 2, axis=1), atol=1e-12)
    np.testing.assert_array_equal(ours.predict_proba(Xt), ours.predict_proba(scores=S))


def test_cross_venn_abers_improves_calibration():
    from sklearn.naive_bayes import GaussianNB
    import pyconformal as pc

    rng = np.random.default_rng(1)
    raw, cross, single = [], [], []
    ll = lambda p, y: -np.mean(np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-12, 1)))
    for rep in range(25):
        X, y = _cva_data(rng, 400)
        X = np.column_stack([X, X[:, :3] + 0.01 * rng.normal(size=(400, 3))])  # duplicated features: overconfident NB
        Xt, yt = _cva_data(rng, 3000)
        Xt = np.column_stack([Xt, Xt[:, :3]])
        raw.append(ll(GaussianNB().fit(X, y).predict_proba(Xt)[:, 1], yt))
        cross.append(ll(pc.CrossVennAbers(GaussianNB(), cv=5, random_state=rep).fit(X, y).predict_proba(Xt)[:, 1], yt))
        single.append(ll(pc.VennAbers(GaussianNB()).fit(X[:200], y[:200]).calibrate(X[200:], y[200:]).predict_proba(Xt)[:, 1], yt))
    assert np.mean(cross) < np.mean(raw)
    assert np.mean(cross) < np.mean(single)  # every row both trains and calibrates


def test_cross_venn_abers_errors():
    import pyconformal as pc

    rng = np.random.default_rng(2)
    s, y, f = rng.uniform(size=90), rng.integers(0, 2, 90), np.arange(90) % 3
    m = pc.CrossVennAbers().calibrate(scores=s, y=y, folds=f)
    out = m.predict_proba(scores=rng.uniform(size=(7, 3)))
    assert out.shape == (7, 2) and np.allclose(out.sum(1), 1) and set(m.predict(scores=rng.uniform(size=(7, 3)))) <= {0, 1}
    assert m.predict_pairs(scores=rng.uniform(size=(7, 3))).shape == (7, 3, 2)
    with pytest.raises(ValueError):
        m.predict_proba(scores=rng.uniform(size=(7, 2)))
    with pytest.raises(TypeError):
        pc.CrossVennAbers().fit(s[:, None], y)
    with pytest.raises(ValueError):
        pc.CrossVennAbers().calibrate(scores=s, y=y, folds=np.zeros(90))
    with pytest.raises(ValueError):
        pc.CrossVennAbers().calibrate(scores=s, y=np.arange(90) % 3, folds=f)
    with pytest.raises(RuntimeError):
        pc.CrossVennAbers().predict_proba(scores=s[:, None])
