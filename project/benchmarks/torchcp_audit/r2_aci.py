"""TorchCP 1.2.1 ACIPredictor and AgACIPredictor against Gibbs & Candes ACI on the same sequences.
Fixed model (prediction = x0), ABS score, lookback 200, the loop of ACIPredictor.evaluate with retraining off.
Reference: alpha_{t+1} = alpha_t + gamma (alpha - err_t), no clipping; level <= 0 gives an infinite interval, level >= 1 an empty one; same 200-step window.
Gibbs & Candes bound on |miss rate - alpha|: (max(alpha, 1 - alpha) + gamma) / (T gamma)."""
import sys, math, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torchcp.regression.score import ABS
from torchcp.regression.predictor import ACIPredictor, AgACIPredictor
R = int(sys.argv[1]); T = 3000; L = 200; alpha = 0.1
class Mean(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x): return x[:, :1]
def errors(kind, g, n):
    z = g.normal(size=n); t = np.arange(n)
    return {"i.i.d.": z, "series trends, model fixed (error grows 0.02 a step)": 0.02 * t + z, "random-walk error": np.cumsum(z) * 0.3,
            "noise scale grows 1% a step": z * 1.01 ** t, "noise scale x5 at step 1500": z * np.where(t < L + 1500, 1, 5)}[kind]
def run_torchcp(p, x, y):
    p.train_indicate = True; p._model.eval(); xl, yl = x[:L], y[:L]
    p.scores = p.calculate_score(p._model(xl).float(), yl); p.q_hat = p._calculate_conformal_value(p.scores, alpha); p.alpha_t = alpha
    pil = p.generate_intervals(p._model(xl).float(), p.q_hat); miss = []; width = []; amin = 1.0
    for t in range(L, len(y)):
        xt, yt = x[t:t + 1], y[t:t + 1]
        with torch.no_grad(): iv = p.predict(xt, xl, yl, pil, train=False, update_alpha=True)
        lo, hi = float(iv[0, 0, 0]), float(iv[0, 0, 1]); miss.append(not (lo <= float(yt) <= hi)); width.append(hi - lo)
        if isinstance(p.alpha_t, float): amin = min(amin, p.alpha_t)
        pil = torch.cat([pil, iv.detach()], 0)[-L:]; xl = torch.cat([xl, xt])[-L:]; yl = torch.cat([yl, yt])[-L:]
    return np.mean(miss), np.mean(np.isinf(width)), amin
def run_reference(e, gamma):
    a = alpha; s = np.abs(e); miss = []; inf = 0
    for t in range(L, len(e)):
        w = np.sort(s[t - L:t])
        if a <= 0: q = np.inf; inf += 1
        elif a >= 1: q = -np.inf
        else:
            k = math.ceil((L + 1) * (1 - a)); q = np.inf if k > L else w[k - 1]; inf += k > L
        m = s[t] > q; miss.append(m); a += gamma * (alpha - m)
    return np.mean(miss), inf / len(miss)
kinds = ["i.i.d.", "series trends, model fixed (error grows 0.02 a step)", "random-walk error", "noise scale grows 1% a step", "noise scale x5 at step 1500"]
for gamma in (0.005, 0.05):
    print(f"ACIPredictor, gamma {gamma}, target miss rate 0.10, {T} steps; Gibbs-Candes bound on the miss rate: {alpha + (0.9 + gamma) / (T * gamma):.4f}")
    for kind in kinds:
        a, b, c, d, am = [], [], [], [], []
        for r in range(R):
            g = np.random.default_rng(r); e = errors(kind, g, T + L); x = torch.tensor(g.normal(size=(T + L, 1)), dtype=torch.float32); y = x[:, 0] + torch.tensor(e, dtype=torch.float32)
            m, i, mn = run_torchcp(ACIPredictor(ABS(), Mean(), gamma=gamma, alpha=alpha), x, y); a.append(m); b.append(i); am.append(mn); m2, i2 = run_reference(e, gamma); c.append(m2); d.append(i2)
        print(f"  {kind:54s} TorchCP miss {np.mean(a):.4f} (max {np.max(a):.4f}), infinite steps {np.mean(b):.3f}, lowest alpha_t {np.min(am):.4f} | reference miss {np.mean(c):.4f} (max {np.max(c):.4f}), infinite steps {np.mean(d):.3f}", flush=True)
print("AgACIPredictor, gamma_list [0.001, 0.01, 0.05, 0.1], default settings")
rec = []
def agg(x, dim): v = torch.mean(x, dim=dim); rec.append(v.flatten().tolist()); return v
for kind in kinds:
    a = []; same = True
    for r in range(max(3, R // 3)):
        g = np.random.default_rng(r); e = errors(kind, g, T + L); x = torch.tensor(g.normal(size=(T + L, 1)), dtype=torch.float32); y = x[:, 0] + torch.tensor(e, dtype=torch.float32)
        p = AgACIPredictor(ABS(), Mean(), gamma_list=[0.001, 0.01, 0.05, 0.1], aggregation_function=agg, alpha=alpha); rec.clear(); m, i, _ = run_torchcp(p, x, y); a.append(m)
        W = np.array(rec).reshape(-1, 4, 2); same &= bool(np.all(W == W[:, :1, :]))
    print(f"  {kind:54s} miss {np.mean(a):.4f} (max {np.max(a):.4f}); the four experts got identical weights at every step: {same}", flush=True)
