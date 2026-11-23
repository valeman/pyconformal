"""TorchCP 1.2.1 graph module: split coverage with APS, DAPS and SNAPS scores and NAPSPredictor on a random two-block graph. Calibration and test nodes are a random split, so node exchangeability holds."""
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torch_geometric.data import Data
from torchcp.classification.score import APS
from torchcp.graph.score import DAPS, SNAPS
from torchcp.graph.predictor import SplitPredictor, NAPSPredictor
R = int(sys.argv[1]); N, K = 1500, 5
def graph(g):
    y = g.integers(0, K, N); P = np.where(y[:, None] == y[None, :], 0.02, 0.002); A = np.triu(g.random((N, N)) < P, 1); i, j = np.nonzero(A)
    ei = torch.tensor(np.r_[np.c_[i, j], np.c_[j, i]].T, dtype=torch.long); a = np.full((N, K), 0.5); a[np.arange(N), y] += 1.5; Pr = np.array([g.dirichlet(r) for r in a])
    return Data(x=torch.tensor(Pr, dtype=torch.float32), edge_index=ei, y=torch.tensor(y), num_nodes=N), torch.tensor(np.log(Pr + 1e-30), dtype=torch.float32)
for name in ("APS", "DAPS", "SNAPS", "NAPSPredictor"):
    cov, sz, err = [], [], None
    for r in range(R):
        g = np.random.default_rng(r); torch.manual_seed(r); d, logits = graph(g); perm = torch.tensor(g.permutation(N)); cal, test = perm[:500], perm[500:]; mask = torch.nn.functional.one_hot(d.y, K).bool()
        try:
            if name == "NAPSPredictor":
                d.test_mask = torch.ones(N, dtype=torch.bool); p = NAPSPredictor(d, alpha=0.1, cutoff=20); idx = torch.arange(N); nodes, S = p.predict_with_logits(logits, idx); yy = d.y[nodes]; cov.append(S[torch.arange(len(nodes)), yy].float().mean().item()); sz.append(S.float().sum(1).mean().item()); continue
            sc = {"APS": lambda: APS(), "DAPS": lambda: DAPS(d, APS()), "SNAPS": lambda: SNAPS(d, APS())}[name]()
            p = SplitPredictor(d, sc, alpha=0.1); p.calculate_threshold(logits, cal, mask, 0.1); S = p.predict_with_logits(logits, test)
        except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:200]}"; break
        cov.append(S[torch.arange(len(test)), d.y[test]].float().mean().item()); sz.append(S.float().sum(1).mean().item())
    print(f"  {name:14s} " + (f"ERROR {err}" if err else f"coverage {np.mean(cov):.4f} (se {np.std(cov) / np.sqrt(R):.4f}), mean set size {np.mean(sz):.2f}"), flush=True)
