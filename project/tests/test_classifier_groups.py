import numpy as np
import pytest

from pyconformal import SplitConformalClassifier


def _draw(rng, n):
    g = rng.integers(0, 3, n)
    conc = np.array([8.0, 2.0, 0.6])[g]  # category 2 is much harder
    y = rng.integers(0, 4, n)
    a = np.full((n, 4), 1.0)
    a[np.arange(n), y] += conc
    p = np.array([rng.dirichlet(r) for r in a])
    y = np.array([rng.choice(4, p=r) for r in p])
    return p, y, g


@pytest.mark.parametrize("score", ["lac", "aps", "raps", "saps"])
def test_coverage_within_each_group(score):
    cov = np.zeros((40, 3))
    plain = np.zeros((40, 3))
    for r in range(40):
        rng = np.random.default_rng(r)
        pc, yc, gc = _draw(rng, 1500)
        pt, yt, gt = _draw(rng, 3000)
        cp = SplitConformalClassifier(alpha=0.1, score=score, random_state=r).calibrate(proba=pc, y=yc, groups=gc)
        hit = cp.predict_set(proba=pt, groups=gt)[np.arange(3000), yt]
        base = SplitConformalClassifier(alpha=0.1, score=score, random_state=r).calibrate(proba=pc, y=yc)
        hit0 = base.predict_set(proba=pt)[np.arange(3000), yt]
        for k in range(3):
            cov[r, k] = hit[gt == k].mean()
            plain[r, k] = hit0[gt == k].mean()
    assert np.all(np.abs(cov.mean(axis=0) - 0.9) < 0.012)
    if score == "lac":  # without groups, coverage is off target inside the categories
        assert np.max(np.abs(plain.mean(axis=0) - 0.9)) > 0.015


def test_group_rules():
    rng = np.random.default_rng(0)
    pc, yc, gc = _draw(rng, 300)
    pt, _, gt = _draw(rng, 20)
    cp = SplitConformalClassifier(alpha=0.1).calibrate(proba=pc, y=yc, groups=gc)
    with pytest.raises(ValueError):
        cp.predict_set(proba=pt)
    with pytest.raises(RuntimeError):
        cp.thresholds()
    assert set(cp.group_thresholds()) == {0, 1, 2}
    unseen = cp.predict_set(proba=pt, groups=np.full(20, 99))
    assert unseen.all()  # a new category gets every class
    nested = cp.predict_set(proba=pt, groups=gt, alpha=[0.2, 0.05])
    assert nested.shape == (20, 4, 2) and np.all(nested[:, :, 0] <= nested[:, :, 1])
    with pytest.raises(ValueError):
        SplitConformalClassifier(class_conditional=True).calibrate(proba=pc, y=yc, groups=gc)
    with pytest.raises(ValueError):
        SplitConformalClassifier().calibrate(proba=pc, y=yc).predict_set(proba=pt, groups=gt)
    strings = SplitConformalClassifier().calibrate(proba=pc, y=yc, groups=np.array(["a", "b", "c"])[gc])
    assert strings.predict_set(proba=pt, groups=np.array(["a", "b", "c"])[gt]).shape == (20, 4)
