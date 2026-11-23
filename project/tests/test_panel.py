import math

import numpy as np
import pytest

from pyconformal import WeightedTQA, conformal_quantile


def reference_wtqa(S, St, Xc, Xt, R, alpha, h, gamma, finite=True):
    """Algorithm 1 of Tu and Giesecke (arXiv:2605.17705), one target at a
    time, written line by line with plain loops."""
    T, N = S.shape
    M = St.shape[1]
    d = Xc.shape[2]
    Q = np.empty((T, M))
    for m in range(M):
        a = alpha                                   # line 1
        mu = np.zeros((N + 1, d))
        w = np.full(N + 1, 1.0 / (N + 1))
        for t in range(T):
            level = max(a, w[N]) if finite else a   # line 6
            aug = np.append(S[t], np.inf)           # line 5
            order = np.argsort(aug, kind="stable")
            q, acc = np.inf, 0.0
            if level >= 1:
                q = -np.inf
            else:
                for k in order:
                    acc += w[k]
                    if acc >= 1 - level - 1e-12:
                        q = aug[k]
                        break
            if finite and a < w[N]:
                q = S[t].max()                      # Lemma B.1
            Q[t, m] = q
            if R[t, m]:                             # line 4, applied at the end of the round
                a = a + gamma * (alpha - float(St[t, m] > q))
            x = np.vstack([Xc[t], Xt[t, m][None, :]])
            mu = (1 - 1 / (t + 1)) * mu + x / (t + 1)   # line 8
            s = mu.std(axis=0)                      # line 9
            s[s == 0] = 1.0
            kap = np.array([math.exp(-np.sum(((mu[k] - mu[N]) / s) ** 2) / (2 * d * h * h)) for k in range(N)] + [1.0])
            w = kap / kap.sum()                     # line 10
    return Q


def _panel(rng, T=120, N=40, M=6, d=3):
    prof = rng.normal(size=(N + M, d))
    X = prof[None] + 0.5 * rng.normal(size=(T, N + M, d))
    scale = 0.5 + np.abs(prof[:, 0])
    S = np.abs(rng.normal(size=(T, N + M))) * scale
    return S[:, :N], S[:, N:], X[:, :N], X[:, N:]


@pytest.mark.parametrize("h,gamma,finite", [(0.6, 0.01, True), (0.3, 0.05, True), (0.6, 0.05, False), (0.4, 0.2, False), (2.0, 0.0, True)])
def test_matches_line_by_line_reference(h, gamma, finite):
    rng = np.random.default_rng(0)
    S, St, Xc, Xt = _panel(rng)
    R = rng.random((120, 6)) < 0.4
    res = WeightedTQA(alpha=0.1, bandwidth=h, gamma=gamma, finite=finite).run(
        scores_cal=S, scores_target=St, X_cal=Xc, X_target=Xt, revealed=R)
    ref = reference_wtqa(S, St, Xc, Xt, R, 0.1, h, gamma, finite)
    assert np.array_equal(res.thresholds, ref)


def test_revealed_round_identity():
    """Proposition 5.5: mean miss on revealed rounds - alpha
    = (alpha - alpha_final) / (gamma * n_revealed), exactly."""
    rng = np.random.default_rng(1)
    S, St, Xc, Xt = _panel(rng, T=400)
    St = St * 3                                     # targets much harder than peers
    R = rng.random((400, 6)) < 0.5
    n = R.sum(axis=0)
    out = {}
    for finite in (False, True):
        tqa = WeightedTQA(alpha=0.1, gamma=0.02, finite=finite)
        res = tqa.run(scores_cal=S, scores_target=St, X_cal=Xc, X_target=Xt, revealed=R)
        miss = (~res.covered & R).sum(axis=0) / n
        assert np.allclose(miss - 0.1, (0.1 - tqa.alpha_t_) / (0.02 * n), atol=1e-10)
        out[finite] = miss
    # With infinite sets allowed the level cannot fall below -gamma (1 - alpha),
    # so the miss rate on revealed rounds is bounded for any data.
    assert np.all(out[False] <= 0.1 + (0.1 + 0.02 * 0.9) / (0.02 * n) + 1e-9)
    # The paper's finitization gives that bound up: targets harder than every peer are missed.
    assert out[True].max() > 0.4


def test_equal_weights_fixed_level_is_split_conformal():
    rng = np.random.default_rng(2)
    S = np.abs(rng.normal(size=(50, 99)))
    St = np.abs(rng.normal(size=(50, 4)))
    res = WeightedTQA(alpha=0.1, bandwidth=np.inf, gamma=0.0).run(scores_cal=S, scores_target=St)
    for t in range(50):
        assert np.all(res.thresholds[t] == conformal_quantile(S[t], 0.1))
    few = WeightedTQA(alpha=0.1, bandwidth=np.inf, gamma=0.0).run(scores_cal=S[:, :5], scores_target=St)
    assert np.all(np.isinf(few.thresholds))         # 5 peers cannot give a 90% set
    fin = WeightedTQA(alpha=0.1, bandwidth=np.inf, gamma=0.0, finite=True).run(scores_cal=S[:, :5], scores_target=St)
    assert np.all(fin.thresholds == S[:, :5].max(axis=1)[:, None]) and fin.fallback.all()


def test_exchangeable_coverage_and_hard_units():
    cov, tail_w, tail_s = [], [], []
    for r in range(8):
        rng = np.random.default_rng(r)
        T, N, M = 300, 200, 40
        hard = rng.random(N + M) < 0.2              # one unit in five is three times noisier
        X = np.where(hard, 2.0, 0.0)[None, :, None] + rng.normal(size=(T, N + M, 2))
        S = np.abs(rng.normal(size=(T, N + M))) * np.where(hard, 3.0, 1.0)
        kw = dict(scores_cal=S[:, :N], scores_target=S[:, N:], X_cal=X[:, :N], X_target=X[:, N:],
                  revealed=rng.random(T) < 0.2)
        w = WeightedTQA(alpha=0.1).run(**kw)
        s = WeightedTQA(alpha=0.1, bandwidth=np.inf, gamma=0.0).run(**kw)
        cov.append(s.coverage); tail_w.append(w.tail_coverage()); tail_s.append(s.tail_coverage())
    assert abs(np.mean(cov) - 0.9) < 0.03           # split conformal is about right on average
    assert np.mean(tail_s) < 0.75                   # but fails the hard units
    assert np.mean(tail_w) > np.mean(tail_s) + 0.2 and np.mean(tail_w) > 0.85


def test_missing_values_intervals_and_online_use():
    rng = np.random.default_rng(3)
    S, St, Xc, Xt = _panel(rng, T=60)
    S2 = S.copy(); S2[5, :10] = np.nan; S2[7] = np.nan
    St2 = St.copy(); St2[9, 0] = np.nan
    res = WeightedTQA(finite=True).run(scores_cal=S2, scores_target=St2, X_cal=Xc, X_target=Xt)
    assert np.isinf(res.thresholds[7]).all() and np.isfinite(res.thresholds[5]).all()
    assert not res.covered[9, 0] and 0.5 < res.coverage <= 1
    P = rng.normal(size=(60, 40)); Pt = rng.normal(size=(60, 6))
    r2 = WeightedTQA().run(y_pred_cal=P, y_cal=P + S, y_pred_target=Pt, y_target=Pt + St, X_cal=Xc, X_target=Xt)
    r1 = WeightedTQA().run(scores_cal=np.abs((P + S) - P), scores_target=np.abs((Pt + St) - Pt), X_cal=Xc, X_target=Xt)
    assert np.array_equal(r2.thresholds, r1.thresholds) and r2.intervals.shape == (60, 6, 2)
    live = WeightedTQA().start(n_cal=40, n_targets=6, n_features=3)
    full = WeightedTQA().run(scores_cal=S, scores_target=St, X_cal=Xc, X_target=Xt)
    for t in range(60):
        q, _ = live.threshold(S[t])
        assert np.array_equal(q, full.thresholds[t])
        live.update(Xc[t], Xt[t], missed=St[t] > q)
    assert live.weights().shape == (6, 41) and np.allclose(live.weights().sum(axis=1), 1)
    with pytest.raises(ValueError):
        WeightedTQA(bandwidth=0)
