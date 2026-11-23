"""TorchCP 1.2.1: ClusteredPredictor fallback, RC3P direct call, NorABS neighbours, EnsemblePredictor, KNN score."""
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
from torchcp.classification.score import LAC, APS, KNN
from torchcp.classification.predictor import SplitPredictor as CSplit, ClusteredPredictor, RC3PPredictor, ClassConditionalPredictor
from torchcp.regression.score import ABS, NorABS
from torchcp.regression.predictor import SplitPredictor, EnsemblePredictor
R = int(sys.argv[1])
print("(a) ClusteredPredictor(LAC), 90% target: share of classes whose threshold is infinite (always in every set)")
def cdraw(g, n, K, pri):
    y = g.choice(K, n, p=pri); A = np.full((n, K), 0.3); A[np.arange(n), y] += 3.0; P = np.array([g.dirichlet(a) for a in A]); return torch.tensor(np.log(P + 1e-30), dtype=torch.float32), torch.tensor(y)
for K, n, label, pri in ((4, 2000, "4 balanced classes, 2,000 calibration rows", None), (10, 5000, "10 balanced classes, 5,000 rows", None), (100, 10000, "100 balanced classes, 10,000 rows", None),
                         (100, 10000, "100 classes, 30 of them rare (about 8 rows each), 10,000 rows", "rare")):
    pri = np.full(K, 1 / K) if pri is None else np.r_[np.full(70, 0.976 / 70), np.full(30, 0.024 / 30)]
    inf, cov, sz, szs, covs = [], [], [], [], []
    for r in range(R):
        g = np.random.default_rng(r); torch.manual_seed(r); Lc, yc = cdraw(g, n, K, pri); Lt, yt = cdraw(g, 3000, K, pri)
        p = ClusteredPredictor(LAC(), alpha=0.1); p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt).int(); inf.append(torch.isinf(p.q_hat).float().mean().item())
        cov.append(S[torch.arange(3000), yt].float().mean().item()); sz.append(S.float().sum(1).mean().item())
        s = CSplit(LAC(), alpha=0.1); s.calculate_threshold(Lc, yc); S2 = s.predict_with_logits(Lt).int(); szs.append(S2.float().sum(1).mean().item()); covs.append(S2[torch.arange(3000), yt].float().mean().item())
    print(f"  {label}: classes with infinite threshold {np.mean(inf):.2f}; coverage {np.mean(cov):.4f}, mean set size {np.mean(sz):.2f} of {K} | plain SplitPredictor: coverage {np.mean(covs):.4f}, size {np.mean(szs):.2f}", flush=True)
print("(b) RC3PPredictor.calculate_threshold called directly (as SplitPredictor allows)")
g = np.random.default_rng(0); Lc, yc = cdraw(g, 500, 4, np.full(4, .25))
try: RC3PPredictor(LAC(), alpha=0.1).calculate_threshold(Lc, yc); print("  ok")
except Exception as e: print(f"  {type(e).__name__}: {e}")
class M2(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return torch.stack([x[:, 0], 0.3 + x[:, 1].abs()], 1)
print("(c0) NorABS + SplitPredictor.predict on batches of different size (model returns mean and spread, shape (n, 2), as documented)")
g = np.random.default_rng(0)
for est in ("variance", "knn_distance"):
    for nb_ in (1, 2, 64):
        xn, yn = None, None
        x_, y_ = torch.tensor(g.normal(size=(300, 2)), dtype=torch.float32), torch.tensor(g.normal(size=300), dtype=torch.float32); dl = DataLoader(TensorDataset(x_, y_), batch_size=300)
        sc = NorABS(data_loader=dl, estimate_type=est, k=5); m = M2(); sc.calibrate(m); p = SplitPredictor(sc, m, alpha=0.1); p.calibrate(dl)
        try: print(f"  {est}, batch of {nb_}: ok, output shape {tuple(p.predict(x_[:nb_]).shape)}")
        except Exception as e: print(f"  {est}, batch of {nb_}: {type(e).__name__}: {' '.join(str(e).split())[:150]}")
print("(c) NorABS(knn_*), 90% target (coverage computed from the scores, because predict fails, see c0): neighbours taken from a separate set (as the example does) or from the calibration set (as the docstring says: 'Calibration dataset')")
class Mean(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return x[:, :1]
def rdraw(g, n):
    x = torch.tensor(g.normal(size=(n, 2)), dtype=torch.float32); y = x[:, 0] + (0.3 + x[:, 1].abs()) * torch.tensor(g.normal(size=n), dtype=torch.float32); return x, y
for est in ("knn_residual", "knn_label", "knn_distance"):
    for k in (5, 20):
        out = {}
        for src in ("separate set", "calibration set"):
            cov = []
            for r in range(R):
                g = np.random.default_rng(r); xn, yn = rdraw(g, 500); xc, yc = rdraw(g, 500); xt, yt = rdraw(g, 2000)
                cal = DataLoader(TensorDataset(xc, yc), batch_size=500); nb = cal if src == "calibration set" else DataLoader(TensorDataset(xn, yn), batch_size=500)
                sc = NorABS(data_loader=nb, estimate_type=est, k=k); m = Mean(); sc.calibrate(m); p = SplitPredictor(sc, m, alpha=0.1); p.calibrate(cal); cov.append((p.calculate_score(m(xt), yt, xt).flatten() <= p.q_hat.flatten()[0]).float().mean().item())
            out[src] = (np.mean(cov), np.std(cov) / np.sqrt(R))
        print(f"  {est:13s} k={k:2d}: " + "; ".join(f"{s} {m:.4f} (se {e:.4f}){' UNDER' if m < 0.9 - 3 * e else ''}" for s, (m, e) in out.items()), flush=True)
print("(d) KNN score with SplitPredictor, 90% target, train features separate from calibration")
cov = []
for r in range(R):
    g = np.random.default_rng(r); mk = lambda n: (lambda y: (torch.tensor(g.normal(size=(n, 2)) + 1.5 * np.c_[np.cos(y * 2.1), np.sin(y * 2.1)], dtype=torch.float32), torch.tensor(y)))(g.integers(0, 3, n))
    Ftr, ytr = mk(300); Fc, yc = mk(500); Ft, yt = mk(1000); p = CSplit(KNN(Ftr, ytr, 3, k=5), alpha=0.1); p.calculate_threshold(Fc, yc); S = p.predict_with_logits(Ft); cov.append(S[torch.arange(1000), yt].float().mean().item())
print(f"  coverage {np.mean(cov):.4f} (se {np.std(cov) / np.sqrt(R):.4f})")
print("(e) EnsemblePredictor (EnbPI), i.i.d. data, linear model, 300 training rows, 5 models, 90% target")
cov = []; nsc = []
for r in range(min(R, 20)):
    g = np.random.default_rng(r); torch.manual_seed(r); xn, yn = rdraw(g, 300); xt, yt = rdraw(g, 2000)
    p = EnsemblePredictor(ABS(), torch.nn.Linear(2, 1), alpha=0.1); p.train(DataLoader(TensorDataset(xn, yn), batch_size=100), ensemble_num=5, subset_num=200, epochs=30, verbose=False)
    iv, _ = p.predict(xt); iv = iv[:, 0, :]; cov.append(((yt >= iv[:, 0]) & (yt <= iv[:, 1])).float().mean().item()); nsc.append(len(p.scores))
print(f"  coverage {np.mean(cov):.4f} (se {np.std(cov) / np.sqrt(len(cov)):.4f}); out-of-bag scores kept {np.mean(nsc):.0f} of 300")
