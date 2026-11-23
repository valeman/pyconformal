"""TorchCP 1.2.1 classification: coverage on i.i.d. data from a perfectly calibrated model, batch-size independence, exact rank, p-values.
Probabilities are passed as log-probabilities, so the default softmax transform returns them unchanged."""
import sys, math, warnings, numpy as np, torch
from fractions import Fraction
warnings.filterwarnings("ignore")
from torchcp.classification.score import LAC, APS, RAPS, SAPS, Margin, TOPK, EntmaxScore
from torchcp.classification.predictor import SplitPredictor, ClassConditionalPredictor, ClusteredPredictor, RC3PPredictor
from torchcp.utils.common import calculate_conformal_value
R = int(sys.argv[1]); K = 10
def verdict(m, se, t): return "UNDER" if m < t - 3 * se else ("over" if m > t + 3 * se + 0.01 else "ok")
def draw(g, n, K=K, conc=0.3):
    P = g.dirichlet(np.ones(K) * conc, n); y = np.array([g.choice(K, p=p) for p in P]); return torch.tensor(np.log(P + 1e-300), dtype=torch.float32), torch.tensor(y)
scores = {"LAC": lambda: LAC(), "APS": lambda: APS(), "APS, not randomized": lambda: APS(randomized=False), "RAPS(penalty=0.01, kreg=2)": lambda: RAPS(penalty=0.01, kreg=2),
          "SAPS(weight=0.2)": lambda: SAPS(weight=0.2), "Margin": lambda: Margin(), "TOPK": lambda: TOPK(), "TOPK, not randomized": lambda: TOPK(randomized=False), "EntmaxScore": lambda: EntmaxScore()}
print("(a) SplitPredictor, 90% target, 1,000 calibration rows; coverage with all test rows in one call and with one row per call")
for name, mk in scores.items():
    cb, co, sz = [], [], []; err = None
    for r in range(R):
        g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc = draw(g, 1000); Lt, yt = draw(g, 600)
        try:
            p = SplitPredictor(mk(), alpha=0.1); p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt)
            O = torch.cat([p.predict_with_logits(Lt[i:i + 1]) for i in range(200)])
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:140]}"; break
        cb.append(S[torch.arange(600), yt].float().mean().item()); co.append(O[torch.arange(200), yt[:200]].float().mean().item()); sz.append(S.float().sum(1).mean().item())
    if err: print(f"  {name:28s} ERROR {err}"); continue
    m, se = np.mean(cb), np.std(cb) / np.sqrt(R)
    print(f"  {name:28s} one call {m:.4f} (se {se:.4f}) {verdict(m, se, 0.9)}; one row per call {np.mean(co):.4f}; mean set size {np.mean(sz):.2f}", flush=True)
print("(b) small calibration sets, LAC and APS, 90% and 95%")
for name, mk in (("LAC", scores["LAC"]), ("APS", scores["APS"])):
    for n in (9, 19, 20, 49, 99):
        for alpha in (0.1, 0.05):
            cb = []; inf = 0
            for r in range(R * 4):
                g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc = draw(g, n); Lt, yt = draw(g, 300)
                p = SplitPredictor(mk(), alpha=alpha); p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt); cb.append(S[torch.arange(300), yt].float().mean().item()); inf += bool(torch.isinf(torch.as_tensor(p.q_hat)))
            m, se = np.mean(cb), np.std(cb) / np.sqrt(len(cb)); print(f"  {name} n={n:3d} alpha={alpha}: {m:.4f} (se {se:.4f}) {verdict(m, se, 1 - alpha)}; threshold infinite in {inf / (R * 4):.2f} of runs")
print("(c) exact rank of calculate_conformal_value, scores 1..n")
for alpha in (0.2, 0.1, 0.05, 0.025, 0.01):
    c = [0, 0, 0, 0]; ex = []
    for n in range(1, 2001):
        k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
        if k > n: continue
        q = float(calculate_conformal_value(torch.arange(1.0, n + 1), alpha)); i = 0 if q == k else (3 if math.isinf(q) else (1 if q > k else 2)); c[i] += 1
        if i and len(ex) < 6: ex.append((n, q, k))
    print(f"  alpha {alpha}: exact {c[0]}, one rank too high {c[1]}, TOO LOW {c[2]}, infinite although a finite rank exists {c[3]}; examples (n, used, exact) {ex}")
print("(d) class-conditional, clustered and RC3P predictors: coverage inside each class; 4 classes of very different difficulty")
def cdraw(g, n):
    y = g.choice(4, n, p=[0.5, 0.3, 0.12, 0.08]); A = np.full((n, 4), 1.0); A[np.arange(n), y] += np.array([8.0, 3.0, 1.2, 0.6])[y]; P = np.array([g.dirichlet(a) for a in A])
    return torch.tensor(np.log(P), dtype=torch.float32), torch.tensor(y)
for pname, mk in (("SplitPredictor", lambda s: SplitPredictor(s, alpha=0.1)), ("ClassConditionalPredictor", lambda s: ClassConditionalPredictor(s, alpha=0.1)),
                  ("ClusteredPredictor", lambda s: ClusteredPredictor(s, alpha=0.1)), ("RC3PPredictor", lambda s: RC3PPredictor(s, alpha=0.1))):
    for sname in ("LAC", "APS"):
        per = [[] for _ in range(4)]; allc = []; sz = []; err = None
        for r in range(max(40, R // 2)):
            g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc = cdraw(g, 2000); Lt, yt = cdraw(g, 3000)
            try:
                p = mk(scores[sname]())
                if pname == "RC3PPredictor": p.num_classes = 4
                p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt).int()
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"; break
            hit = S[torch.arange(3000), yt].float().numpy(); allc.append(hit.mean()); sz.append(S.float().sum(1).mean().item())
            for c_ in range(4): per[c_].append(hit[yt.numpy() == c_].mean())
        if err: print(f"  {pname:26s} {sname}: ERROR {err}"); continue
        print(f"  {pname:26s} {sname}: overall {np.mean(allc):.4f}; per class {[round(float(np.mean(x)), 3) for x in per]}; mean set size {np.mean(sz):.2f}", flush=True)
print("(e) p-values of the true label (SplitPredictor.predict_p needs a model; computed here through an identity model)")
class Id(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return x
for smooth in (True, False):
    ps = []
    for r in range(R):
        g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc = draw(g, 200); Lt, yt = draw(g, 200)
        p = SplitPredictor(LAC(), model=Id(), alpha=0.1); p.calculate_threshold(Lc, yc); ps.append(p.predict_p(Lt, yt, smooth=smooth).numpy())
    ps = np.concatenate(ps); print(f"  smooth={smooth}: P(p <= 0.1) = {(ps <= 0.1).mean():.4f}, P(p <= 0.05) = {(ps <= 0.05).mean():.4f}, mean {ps.mean():.4f}")
print("(f) does a score change the caller's tensor?")
for name, mk in (("Margin(score_type='identity')", lambda: Margin(score_type="identity")), ("LAC(score_type='identity')", lambda: LAC(score_type="identity")), ("APS(score_type='identity')", lambda: APS(score_type="identity")), ("SAPS(score_type='identity')", lambda: SAPS(score_type="identity")), ("Margin() with softmax", lambda: Margin())):
    g = np.random.default_rng(0); P = torch.tensor(g.dirichlet(np.ones(5), 6), dtype=torch.float32); y = torch.tensor(g.integers(0, 5, 6)); before = P.clone(); mk()(P, y); a = bool(torch.equal(P, before)); before = P.clone(); mk()(P); b = bool(torch.equal(P, before))
    print(f"  {name:32s}: input unchanged after scoring with labels: {a}; after scoring all labels: {b}" + ("" if a else f"   e.g. row 0 now {P[0].numpy().round(3).tolist()}"))
