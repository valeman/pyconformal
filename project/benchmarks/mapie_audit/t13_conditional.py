"""Conditional split conformal (Gibbs, Cherian, Candes): with a feature map of group indicators, coverage must be
at least the level inside every group, not just on average. Two groups with very different noise."""
import sys, time, numpy as np, warnings
from common import *
warnings.simplefilter("ignore")
from mapie.conditional_conformal_prediction import ConditionalSplitConformalRegressor, ConditionalSplitConformalClassifier
from mapie.regression import SplitConformalRegressor
R, ncal, ntest = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
phi = lambda X: np.column_stack([np.asarray(X)[:, 0] == 0, np.asarray(X)[:, 0] == 1]).astype(float)
def draw(n, rng):
    g = (rng.random(n) < 0.3).astype(float); y = rng.normal(size=n) * np.where(g == 1, 5.0, 1.0)
    return g.reshape(-1, 1), y
for label, kw in (("exact=True, randomize=False (defaults)", {}), ("exact=True, randomize=True", {"randomize": True}), ("exact=False", {"exact": False})):
    cov = {0: [], 1: [], "all": []}; t0 = time.time(); err = None; inf = 0
    for r in range(R if kw.get("exact", True) else R // 5):
        rng = np.random.default_rng(86028121 * r + 5); Xc, yc = draw(ncal, rng); Xt, yt = draw(ntest, rng)
        try:
            m = ConditionalSplitConformalRegressor(phi, Const(0.0), confidence_level=0.9, prefit=True, seed=r, **kw).conformalize(Xc, yc)
            iv = np.asarray(m.predict_interval(Xt)[1]); iv = iv[:, :, 0] if iv.ndim == 3 else iv
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:140]}"; break
        hit = (iv[:, 0] <= yt) & (yt <= iv[:, 1]); inf += np.isinf(iv).any(axis=1).mean()
        for g in (0, 1):
            if (Xt[:, 0] == g).any(): cov[g].append(hit[Xt[:, 0] == g].mean())
        cov["all"].append(hit.mean())
    if err: print(f"conditional regressor, {label}: ERROR {err}"); continue
    out = []
    for k in (0, 1, "all"):
        m_, se = summary(cov[k]); out.append(f"{'group ' + str(k) if k != 'all' else 'overall'} {m_:.4f} (se {se:.4f}) {verdict(m_, se, 0.9)}")
    print(f"conditional regressor, {label}: " + "; ".join(out) + f"; share infinite {inf/R:.3f}  ({time.time()-t0:.0f}s)", flush=True)
# the ordinary split regressor on the same data, for contrast
cov = {0: [], 1: []}
for r in range(R):
    rng = np.random.default_rng(86028121 * r + 5); Xc, yc = draw(ncal, rng); Xt, yt = draw(ntest, rng)
    iv = SplitConformalRegressor(Const(0.0), confidence_level=0.9, prefit=True).conformalize(Xc, yc).predict_interval(Xt)[1][:, :, 0]
    hit = (iv[:, 0] <= yt) & (yt <= iv[:, 1])
    for g in (0, 1): cov[g].append(hit[Xt[:, 0] == g].mean())
print(f"ordinary split regressor: group 0 {np.mean(cov[0]):.4f}; group 1 {np.mean(cov[1]):.4f}")
# classifier
def drawc(n, rng):
    g = (rng.random(n) < 0.3).astype(int); K = 4
    conc = np.where(g[:, None] == 1, 0.6, 6.0)            # group 1: model far less sure
    P = rng.dirichlet(np.ones(K), n) ** conc; P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in P])
    return np.column_stack([P, g]), y, g
phic = lambda X: np.column_stack([np.asarray(X)[:, 4] == 0, np.asarray(X)[:, 4] == 1]).astype(float)
for score in ("lac", "aps"):
    cov = {0: [], 1: [], "all": []}; err = None; t0 = time.time()
    for r in range(R):
        rng = np.random.default_rng(15487469 * r + 3); Xc, yc, gc = drawc(ncal, rng); Xt, yt, gt = drawc(ntest, rng)
        try:
            m = ConditionalSplitConformalClassifier(phic, FixedClf(4).fit(None), confidence_level=0.9, conformity_score=score, prefit=True, seed=r).conformalize(Xc, yc)
            S = np.asarray(m.predict_set(Xt)[1]); S = S[:, :, 0] if S.ndim == 3 else S
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:140]}"; break
        hit = S[np.arange(len(yt)), yt].astype(bool)
        for g in (0, 1): cov[g].append(hit[gt == g].mean())
        cov["all"].append(hit.mean())
    if err: print(f"conditional classifier, {score}: ERROR {err}"); continue
    out = []
    for k in (0, 1, "all"):
        m_, se = summary(cov[k]); out.append(f"{'group ' + str(k) if k != 'all' else 'overall'} {m_:.4f} (se {se:.4f}) {verdict(m_, se, 0.9)}")
    print(f"conditional classifier, {score}: " + "; ".join(out) + f"  ({time.time()-t0:.0f}s)", flush=True)
