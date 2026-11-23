"""TorchCP 1.2.1 regression scores with SplitPredictor. Fixed models (no training), so any coverage gap is the conformal step.
Data: y = x0 + s(x) * N(0,1), s = 0.5 + |x1|. The 'quantile model' returns bands that are deliberately too narrow (z = 0.8 instead of 1.645), so the raw band covers about 58%."""
import sys, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
from torchcp.regression.score import ABS, CQR, CQRR, CQRM, CQRFM
from torchcp.regression.predictor import SplitPredictor
R = int(sys.argv[1])
class Base(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
class Mean(Base):
    def forward(self, x): return x[:, :1]
class Q2(Base):
    def forward(self, x): s = 0.5 + x[:, 1].abs(); return torch.stack([x[:, 0] - 0.8 * s, x[:, 0] + 0.8 * s], 1)
class Q3(Base):
    def forward(self, x): s = 0.5 + x[:, 1].abs(); return torch.stack([x[:, 0] - 0.8 * s, x[:, 0], x[:, 0] + 0.8 * s], 1)
def draw(g, n):
    x = torch.tensor(g.normal(size=(n, 2)), dtype=torch.float32); y = x[:, 0] + (0.5 + x[:, 1].abs()) * torch.tensor(g.normal(size=n), dtype=torch.float32); return x, y
print("90% target, 1,000 calibration rows, 2,000 test rows")
for name, sc, model in (("ABS", ABS, Mean), ("CQR", CQR, Q2), ("CQRR", CQRR, Q2), ("CQRM", CQRM, Q3), ("CQRFM", CQRFM, Q3)):
    cov, wid, raw, qs = [], [], [], []
    for r in range(R):
        g = np.random.default_rng(r); xc, yc = draw(g, 1000); xt, yt = draw(g, 2000)
        p = SplitPredictor(sc(), model(), alpha=0.1); p.calibrate(DataLoader(TensorDataset(xc, yc), batch_size=500)); iv = p.predict(xt)[:, 0, :]
        cov.append(((yt >= iv[:, 0]) & (yt <= iv[:, 1])).float().mean().item()); wid.append((iv[:, 1] - iv[:, 0]).mean().item()); qs.append(float(p.q_hat.flatten()[0]))
        m = model()(xt); raw.append(((yt >= m[:, 0]) & (yt <= m[:, -1])).float().mean().item() if m.shape[1] > 1 else float("nan"))
    m_, se = np.mean(cov), np.std(cov) / np.sqrt(R)
    print(f"  {name:6s} coverage {m_:.4f} (se {se:.4f}) {'UNDER' if m_ < 0.9 - 3 * se else 'ok'}; mean width {np.mean(wid):.3f}; raw band coverage before the conformal step {np.mean(raw):.4f}; mean q_hat {np.mean(qs):.4f}")
print("p-values of the true y (SplitPredictor.predict_p), ABS")
for smooth in (True, False):
    ps = []
    for r in range(R):
        g = np.random.default_rng(r); xc, yc = draw(g, 200); xt, yt = draw(g, 200); p = SplitPredictor(ABS(), Mean(), alpha=0.1); p.calibrate(DataLoader(TensorDataset(xc, yc), batch_size=500)); ps.append(p.predict_p(xt, yt, smooth=smooth).flatten().numpy())
    ps = np.concatenate(ps); print(f"  smooth={smooth}: n={ps.size}, P(p <= 0.1) = {(ps <= 0.1).mean():.4f}, P(p <= 0.05) = {(ps <= 0.05).mean():.4f}")
