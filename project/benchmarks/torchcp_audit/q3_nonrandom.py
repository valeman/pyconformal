"""TorchCP 1.2.1: non-randomised scores when the model is accurate. 10 classes, 5,000 calibration and 20,000 test rows, labels drawn from the probabilities, 90% target."""
import warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torchcp.classification.score import APS, RAPS, SAPS, TOPK
from torchcp.classification.predictor import SplitPredictor
def draw(g, n, boost):
    A = np.full((n, 10), 0.3); A[np.arange(n), g.integers(0, 10, n)] += boost; P = np.array([g.dirichlet(a) for a in A]); y = (P.cumsum(1) > g.random((n, 1))).argmax(1); return torch.tensor(np.log(P + 1e-30), dtype=torch.float32), torch.tensor(y), (P.argmax(1) == y).mean()
print("probs = [0.6, 0.3, 0.1], true label = top class: score used in calibration vs score used to build the set")
pr = torch.tensor([[0.6, 0.3, 0.1]])
for n_, s in (("SAPS", SAPS(score_type="identity", randomized=False)), ("APS", APS(score_type="identity", randomized=False)), ("RAPS", RAPS(score_type="identity", randomized=False)), ("TOPK", TOPK(score_type="identity", randomized=False))):
    print(f"  {n_}(randomized=False): calibration {float(s(pr.clone(), torch.tensor([0]))[0]):.3f}, prediction {float(s(pr.clone())[0, 0]):.3f}")
for boost in (1.5, 20, 40, 150):
    row = []
    for name, mk in (("SAPS", lambda rd: SAPS(weight=0.2, randomized=rd)), ("APS", lambda rd: APS(randomized=rd)), ("RAPS", lambda rd: RAPS(penalty=0.01, kreg=1, randomized=rd)), ("TOPK", lambda rd: TOPK(randomized=rd))):
        for rd in (True, False):
            cov = []
            for r in range(5):
                g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc, acc = draw(g, 5000, boost); Lt, yt, _ = draw(g, 20000, boost); p = SplitPredictor(mk(rd), alpha=0.1); p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt); cov.append(S[torch.arange(20000), yt].float().mean().item())
            row.append(f"{name} {'rand' if rd else 'not rand'} {np.mean(cov):.3f}")
    print(f"  top-1 accuracy {acc:.2f}: " + "; ".join(row), flush=True)
