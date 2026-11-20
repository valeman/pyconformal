import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from pyconformal import APS, LAC, RAPS, SAPS, SplitConformalClassifier
from pyconformal.scores import get_score

pytestmark = pytest.mark.filterwarnings("error")


def brute(score, p, u):
    """Scores of all classes of one sample, written directly from the papers."""
    order = sorted(range(len(p)), key=lambda j: -p[j])
    out = np.empty(len(p))
    for r, j in enumerate(order, start=1):
        before = sum(p[order[i]] for i in range(r - 1))
        if isinstance(score, SAPS):
            top = p[order[0]]
            out[j] = u * top if r == 1 else top + (r - 2 + u) * score.weight
        elif isinstance(score, RAPS):
            out[j] = before + u * p[j] + score.penalty * max(0, r - score.k_reg)
        elif isinstance(score, APS):
            out[j] = before + u * p[j]
        else:
            out[j] = 1 - p[j]
    return out


SCORES = [LAC(), APS(), APS(randomized=False), RAPS(0.1, 1), RAPS(0.05, 2, randomized=False),
          SAPS(), SAPS(0.5, randomized=False)]


@pytest.mark.parametrize("score", SCORES, ids=repr)
def test_scores_match_definitions(score):
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(6), size=200)
    u = rng.random(200)
    got = score.scores(P, u)
    for i in range(200):
        want = brute(score, P[i], u[i] if score.randomized else 1.0)
        assert np.allclose(got[i], want, atol=1e-14)
    assert got.shape == P.shape


def data(rng, n, K=6, signal=2.0):
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K)) * 1.5 + signal * np.eye(K)[y]
    P = np.exp(logits)
    return P / P.sum(1, keepdims=True), y


class Passthrough:
    """A 'model' whose features already are probabilities."""

    def __init__(self, K):
        self.classes_ = np.arange(K)

    def predict_proba(self, X):
        return X

    def predict(self, X):
        return X.argmax(1)


@pytest.mark.parametrize("score", SCORES, ids=repr)
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_coverage(score, alpha):
    n_cal, covs = 99, []
    for r in range(300):
        rng = np.random.default_rng(1000 * r + int(100 * alpha))
        Pc, yc = data(rng, n_cal)
        Pt, yt = data(rng, 500)
        m = SplitConformalClassifier(Passthrough(6), alpha, score=score, random_state=r)
        sets = m.calibrate(Pc, yc).predict_set(Pt)
        covs.append(sets[np.arange(500), yt].mean())
    mean, se = np.mean(covs), np.std(covs, ddof=1) / np.sqrt(len(covs))
    assert mean >= 1 - alpha - 3 * se
    if score.randomized or isinstance(score, LAC):
        # continuous scores: coverage is exactly ceil((n+1)(1-alpha))/(n+1) = 1 - alpha here
        assert mean == pytest.approx(1 - alpha, abs=3 * se + 1e-3)


def test_randomisation_makes_sets_smaller():
    rng = np.random.default_rng(1)
    Pc, yc = data(rng, 2000)
    Pt, _ = data(rng, 5000)
    size = {}
    for name, sc in (("rand", APS()), ("plain", APS(randomized=False))):
        m = SplitConformalClassifier(Passthrough(6), 0.1, score=sc, random_state=0)
        size[name] = m.calibrate(Pc, yc).predict_set(Pt).sum(1).mean()
    assert size["rand"] < size["plain"]


def test_class_conditional_coverage():
    cov = np.zeros((200, 3))
    for r in range(200):
        rng = np.random.default_rng(r)
        # class 2 is much harder than the others
        def make(n):
            y = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
            logits = rng.normal(size=(n, 3)) * 1.5 + np.array([3.0, 3.0, 0.5])[y][:, None] * np.eye(3)[y]
            P = np.exp(logits)
            return P / P.sum(1, keepdims=True), y
        Pc, yc = make(600)
        Pt, yt = make(3000)
        m = SplitConformalClassifier(Passthrough(3), 0.1, score="aps", class_conditional=True, random_state=10_000 + r)  # not the data seed
        sets = m.calibrate(Pc, yc).predict_set(Pt)
        hit = sets[np.arange(3000), yt]
        cov[r] = [hit[yt == k].mean() for k in range(3)]
    assert np.all(cov.mean(0) >= 0.895) and np.all(cov.mean(0) <= 0.915)
    # marginal calibration under-covers the hard class on the same kind of data
    m = SplitConformalClassifier(Passthrough(3), 0.1, score="aps", random_state=7)
    hit = m.calibrate(Pc, yc).predict_set(Pt)[np.arange(3000), yt]
    assert hit[yt == 2].mean() < 0.86
    # a class with no calibration data is always included
    keep = yc != 1
    m = SplitConformalClassifier(Passthrough(3), 0.1, class_conditional=True)
    assert m.calibrate(Pc[keep], yc[keep]).predict_set(Pt)[:, 1].all()


def test_with_real_model_and_api():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, size=1500)
    X = rng.normal(size=(1500, 3)) + 1.2 * np.eye(4)[y][:, :3]
    for score in ("lac", "aps", "raps", "saps", RAPS(penalty=0.1, k_reg=1)):
        m = SplitConformalClassifier(LogisticRegression(), score=score, random_state=0)
        sets = m.fit(X[:500], y[:500]).calibrate(X[500:1000], y[500:1000]).predict_set(X[1000:])
        assert sets.shape == (500, 4) and 0.85 < sets[np.arange(500), y[1000:]].mean() < 0.96
        wide = m.predict_set(X[1000:], alpha=0.01).sum(1).mean()
        assert wide >= sets.sum(1).mean()
    with pytest.raises(ValueError):
        get_score("nope")
    with pytest.raises(ValueError):
        SAPS(weight=0)
    with pytest.raises(ValueError):
        RAPS(penalty=-1)
    assert "RAPS(penalty=0.1" in repr(RAPS(0.1, 1))


@pytest.mark.parametrize("name", ["aps", "raps", "saps"])
@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("K", [7, 300])
def test_compiled_and_numpy_paths_agree(name, randomized, K):
    """The compiled kernels rank only the top classes of each row. They must
    return exactly what the full-sort NumPy code returns, ties included."""
    from pyconformal import scores as S

    if not S.HAVE_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(11)
    n = 400
    proba = rng.dirichlet(np.full(K, 0.3), size=n)
    proba[:5] = 1.0 / K  # every class tied
    proba[5:10, : K // 2] = proba[5:10, K // 2 : 2 * (K // 2)]  # pairs tied
    proba[5:10] /= proba[5:10].sum(axis=1, keepdims=True)
    cols = rng.integers(0, K, size=n)
    u = rng.random(n) if randomized else None
    sc = S.get_score(name)
    sc.randomized = randomized
    thresholds = [0.05, 0.8, 0.999, 3.0, rng.uniform(0.2, 1.1, size=K)]
    mixed = rng.uniform(0.2, 1.1, size=K)
    mixed[::3] = np.inf  # classes with too little calibration data
    thresholds.append(mixed)

    def run():
        return sc.label_scores(proba, cols, u), [sc.sets(proba, q, u) for q in thresholds]

    fast_lab, fast_sets = run()
    S.HAVE_NUMBA = False
    try:
        slow_lab, slow_sets = run()
    finally:
        S.HAVE_NUMBA = True
    np.testing.assert_array_equal(fast_lab, slow_lab)
    for f, sl in zip(fast_sets, slow_sets):
        np.testing.assert_array_equal(f, sl)


def test_parallel_search_matches_numpy():
    from pyconformal._search import search

    rng = np.random.default_rng(5)
    a = np.sort(rng.integers(0, 500, size=3000).astype(float))
    v = rng.integers(-5, 505, size=50_000).astype(float)
    for side in ("left", "right"):
        np.testing.assert_array_equal(search(a, v, side), np.searchsorted(a, v, side=side))
