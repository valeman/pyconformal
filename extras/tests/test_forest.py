import numpy as np
import pytest

import pyconformal as pc
from pyconformal import forest as F

pytestmark = pytest.mark.skipif(not F.HAVE_NUMBA, reason="numba not installed")


def data(rng, n, d=6):
    X = rng.uniform(-1, 1, (n, d))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + (0.3 + 0.5 * np.abs(X[:, 2])) * rng.standard_exponential(n)
    return X, y


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    X, y = data(rng, 4000)
    return pc.QuantileForest(60, random_state=1).fit(X, y), X, y, rng


def leaves_by_hand(m, x):
    """Leaf (start, size) in every tree for one binned row, in plain Python."""
    out = []
    for t in range(m._nodes.shape[0]):
        node = 0
        while m._nodes[t, node] >> 40:
            w = int(m._nodes[t, node])
            node = (w & 0xFFFFFFFF) + int(x[(w >> 40) - 1] > ((w >> 32) & 0xFF))
        out.append((int(m._nodes[t, node]) & 0xFFFFFFFF, int(m._b[t, node])))
    return out


def test_quantiles_match_the_definition(fitted):
    m, X, y, rng = fitted
    Xt, _ = data(rng, 25)
    levels = np.array([0.0, 0.01, 0.1, 0.5, 0.77, 0.9, 1.0])
    got = m.predict(Xt, quantiles=levels)
    mean = m.predict(Xt)
    Xb = m._bin(Xt)
    T = m._nodes.shape[0]
    for i in range(25):
        vals, w = [], []
        for t, (s, c) in enumerate(leaves_by_hand(m, Xb[i])):
            vals += list(m._ys[m._rk[t, s : s + c]])
            w += [1.0 / (c * T)] * c
        vals, w = np.array(vals), np.array(w)
        o = np.argsort(vals, kind="stable")
        cum = np.cumsum(w[o])
        want = [vals[o][min(np.searchsorted(cum, q - 1e-12), len(o) - 1)] for q in levels]
        np.testing.assert_array_equal(got[i], want)
        np.testing.assert_allclose(mean[i], np.sum(vals * w), rtol=1e-12)
        assert got[i, 0] == vals.min() and got[i, -1] == vals.max()


def test_trees_respect_settings(fitted):
    m, X, y, _ = fitted
    is_leaf = (m._nodes >> 40) == 0
    for t in range(m._nodes.shape[0]):
        k = m.n_nodes_[t]
        sizes = m._b[t, :k][is_leaf[t, :k]]
        assert sizes.min() >= 5 and sizes.sum() == len(y)  # every drawn sample is in one leaf
    shallow = pc.QuantileForest(5, max_depth=2, random_state=0).fit(X, y)
    assert shallow.n_nodes_.max() <= 7
    sub = pc.QuantileForest(5, max_samples=0.25, bootstrap=False, random_state=0).fit(X, y)
    assert sub._rk.shape[1] == 1000 and len(np.unique(sub._rk[0])) == 1000


def test_same_seed_same_forest_whatever_the_threads(fitted):
    import numba

    m, X, y, rng = fitted
    Xt, _ = data(rng, 300)
    lv = [0.1, 0.5, 0.9]
    ref = m.predict(Xt, quantiles=lv)
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        again = pc.QuantileForest(60, random_state=1).fit(X, y).predict(Xt, quantiles=lv)
    finally:
        numba.set_num_threads(before)
    np.testing.assert_array_equal(ref, again)
    assert not np.array_equal(ref, pc.QuantileForest(60, random_state=2).fit(X, y).predict(Xt, quantiles=lv))


def test_accuracy_and_shapes(fitted):
    m, X, y, rng = fitted
    Xt, yt = data(rng, 5000)
    lv = np.array([0.95, 0.05, 0.5])  # order given is order returned
    q = m.predict(Xt, quantiles=lv)
    assert q.shape == (5000, 3) and np.all(q[:, 1] <= q[:, 2]) and np.all(q[:, 2] <= q[:, 0])
    assert m.predict(Xt, quantiles=0.5).shape == (5000,)
    cover = ((q[:, 1] <= yt) & (yt <= q[:, 0])).mean()
    assert 0.85 < cover < 0.98
    pin = lambda pred, t: np.mean(np.maximum(t * (yt - pred), (t - 1) * (yt - pred)))
    for j, t in enumerate(lv):
        assert pin(q[:, j], t) < 0.8 * pin(np.quantile(y, t), t)  # far better than ignoring X
    assert np.mean((m.predict(Xt) - yt) ** 2) < 0.7 * np.var(yt)
    # heteroscedastic by construction: wider where |x2| is large
    width = q[:, 0] - q[:, 1]
    assert width[np.abs(Xt[:, 2]) > 0.7].mean() > 1.3 * width[np.abs(Xt[:, 2]) < 0.3].mean()


def test_feeds_the_conformal_methods(fitted):
    m, X, y, rng = fitted
    Xc, yc = data(rng, 1500)
    Xt, yt = data(rng, 4000)
    for cp in (
        pc.ConformalizedQuantileRegressor(m),
        pc.ConformalInterquantileRegressor(m, plus=True, quantile_levels=50),
        pc.ConformalInterquantileRegressor(pc.QuantileForest(30, random_state=0), quantile_levels=50).fit(X, y),
    ):
        iv = cp.calibrate(Xc, yc).predict_interval(Xt)
        assert 0.87 < ((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean() < 0.94
    cti = pc.ConformalThresholdedIntervals(m, quantile_levels=50).calibrate(Xc, yc)
    assert cti.predict_set(Xt).shape == (4000, 52)


def test_edge_cases_and_errors():
    rng = np.random.default_rng(3)
    X, y = data(rng, 50)
    const = pc.QuantileForest(3, random_state=0).fit(X, np.ones(50))
    np.testing.assert_array_equal(const.predict(X, quantiles=[0.1, 0.9]), 1.0)
    flat = pc.QuantileForest(3, random_state=0).fit(np.zeros((50, 2)), y)  # nothing to split on
    assert np.all(flat.predict(np.zeros((4, 2)), quantiles=1.0) == y.max())
    tiny = pc.QuantileForest(3, random_state=0).fit(X[:3], y[:3])
    assert tiny.predict(X[:2], quantiles=[0.5]).shape == (2, 1)
    with pytest.raises(RuntimeError):
        pc.QuantileForest().predict(X)
    with pytest.raises(ValueError):
        const.predict(X, quantiles=[1.5])
    with pytest.raises(ValueError):
        const.predict(X[:, :3])
    bad = X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        pc.QuantileForest(2).fit(bad, y)
    assert pc.QuantileForest(7).set_params(min_samples_leaf=2).get_params()["min_samples_leaf"] == 2


def test_one_sample_per_leaf_mode(fitted):
    m, X, y, rng = fitted
    Xt, yt = data(rng, 4000)
    one = pc.QuantileForest(60, random_state=1, max_samples_leaf=1).fit(X, y)
    lv = np.array([0.05, 0.5, 0.95])
    q1, q = one.predict(Xt, quantiles=lv), m.predict(Xt, quantiles=lv)
    pin = lambda pred: np.mean([np.mean(np.maximum(t * (yt - pred[:, j]), (t - 1) * (yt - pred[:, j]))) for j, t in enumerate(lv)])
    assert pin(q1) < 1.05 * pin(q)  # close to the exact forest
    assert np.all(q1[:, 0] <= q1[:, 1]) and np.all(q1[:, 1] <= q1[:, 2])
    assert np.isin(q1, y).all()  # predictions are training outcomes
    np.testing.assert_array_equal(one.predict(Xt, quantiles=[0.0, 1.0])[:, 0] <= one.predict(Xt), True)
    with pytest.raises(RuntimeError):
        one.predict_oob()
    with pytest.raises(ValueError):
        pc.QuantileForest(3, max_samples_leaf=2).fit(X, y)


def test_out_of_bag_predictions(fitted):
    m, X, y, rng = fitted
    lv = [0.05, 0.5, 0.95]
    oob, insample = m.predict_oob(quantiles=lv), m.predict(X, quantiles=lv)
    cover = lambda q: ((q[:, 0] <= y) & (y <= q[:, 2])).mean()
    assert cover(insample) > cover(oob) + 0.02  # in-sample predictions are too confident
    assert 0.85 < cover(oob) < 0.97
    # by hand for a few rows: only trees that did not see the row vote
    T = m._nodes.shape[0]
    inbag = np.zeros((T, len(y)), dtype=bool)
    np.put_along_axis(inbag, m._order[m._rk], True, axis=1)
    assert 0.3 < 1 - inbag.mean() < 0.42  # about 1/e of the trees per row
    for i in (0, 17, 3999):
        vals, w = [], []
        trees = np.flatnonzero(~inbag[:, i])
        for t, (s, c) in enumerate(leaves_by_hand(m, m._Xb[i])):
            if t in trees:
                vals += list(m._ys[m._rk[t, s : s + c]])
                w += [1.0 / (c * len(trees))] * c
        vals, w = np.array(vals), np.array(w)
        o = np.argsort(vals, kind="stable")
        cum = np.cumsum(w[o])
        np.testing.assert_array_equal(oob[i], [vals[o][min(np.searchsorted(cum, q - 1e-12), len(o) - 1)] for q in lv])
        assert y[i] not in vals or (y == y[i]).sum() > 1
    with pytest.raises(RuntimeError, match="every tree"):
        pc.QuantileForest(3, bootstrap=False, random_state=0).fit(X, y).predict_oob()


def test_conformal_calibration_without_a_calibration_set():
    rng = np.random.default_rng(21)
    cov = {"cqr": [], "cir": []}
    for rep in range(12):
        X, y = data(rng, 1500)
        Xt, yt = data(rng, 3000)
        f = pc.QuantileForest(80, random_state=rep).fit(X, y)
        for name, cp in (("cqr", pc.ConformalizedQuantileRegressor(f)), ("cir", pc.ConformalInterquantileRegressor(f, plus=True, quantile_levels=50))):
            iv = cp.calibrate_oob(y).predict_interval(Xt)
            cov[name].append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean())
    for name, v in cov.items():
        assert 0.885 < np.mean(v) < 0.93, (name, np.mean(v))
    with pytest.raises(TypeError):
        pc.ConformalizedQuantileRegressor().calibrate_oob(y)
