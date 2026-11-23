"""ConditionalExpectedRiskController (torch). Scores uniform on (0,1), 16 labels per row, label true with probability = score.
Prediction keeps labels with score >= parameter. Risk = 1 - recall, target 0.1. With a constant feature map the method has one global parameter.
The ideal global parameter is the one where expected risk = 0.1."""
import sys, time, numpy as np, warnings
warnings.simplefilter("ignore")
import torch
from mapie.risk_control.adaptive_conformal_risk_control import ConditionalExpectedRiskController
def draw(n, rng):
    X = rng.random((n, 16)); y = (rng.random((n, 16)) < X).astype(float); keep = y.sum(1) > 0; return X[keep], y[keep]
def pf(X, p=None):
    X = np.asarray(X)
    return X if p is None else X >= np.asarray(p).reshape(-1, 1)
fm = lambda X: np.ones((1, 1)) if np.asarray(X).ndim == 1 else np.ones((len(X), 1))
Xb, yb = draw(200000, np.random.default_rng(7))
def true_risk(lam): return 1 - (((Xb >= lam) & (yb == 1)).sum(1) / yb.sum(1)).mean()
grid = np.linspace(0, 1, 501); ideal = grid[np.argmin(np.abs([true_risk(l) - 0.1 for l in grid]))]
print(f"ideal global parameter {ideal:.3f} (true risk there {true_risk(ideal):.3f})")
R = int(sys.argv[1])
for label, kw, ep in (("defaults (learning_rate 1e-4, 100 epochs)", {}, 100), ("learning_rate 1e-2, 100 epochs", {"learning_rate": 1e-2}, 100), ("learning_rate 1e-2, 400 epochs", {"learning_rate": 1e-2}, 400)):
    risks = []; params = []; t0 = time.time(); err = None
    for r in range(R):
        rng = np.random.default_rng(r); np.random.seed(r); torch.manual_seed(r); Xc, yc = draw(500, rng); Xt, yt = draw(40, rng)
        try:
            c = ConditionalExpectedRiskController(pf, fm, 0.1, "recall", **kw); c.conformalize(Xc, yc, n_epochs=ep)
            S = np.asarray(c.predict(Xt)); p = np.asarray(c.best_predict_params_)
        except Exception as e:
            err = f"{type(e).__name__}: {' '.join(str(e).split())[:160]}"; break
        params.append(p.mean()); risks.append(np.mean([true_risk(v) for v in p]))
    if err: print(f"{label}: ERROR {err}"); continue
    print(f"{label:44s}: mean parameter {np.mean(params):.3f} (ideal {ideal:.3f}); true risk mean {np.mean(risks):.4f}, highest run {np.max(risks):.4f} (target 0.10); mean share of labels predicted {np.mean([(Xb >= v).mean() for v in params]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
