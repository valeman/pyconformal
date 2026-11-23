"""TorchCP 1.2.1, remaining parts: WeightedPredictor, CDLoss direction, metrics, CPD, R2CCP, LLM risk control."""
import sys, math, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
R = int(sys.argv[1]); part = sys.argv[2] if len(sys.argv) > 2 else "all"
class P(torch.nn.Module):
    def __init__(self): super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1))
if part in ("all", "w"):
    from torchcp.classification.score import LAC
    from torchcp.classification.predictor import WeightedPredictor
    print("(a) WeightedPredictor under covariate shift. Source x ~ N(0,1), target x ~ N(1,1), true weight w(x) = exp(x - 0.5) given to it through the domain classifier. 3 classes, LAC, 90% target.")
    class Enc(P):
        def forward(self, x): return x[:, :1]
    class Dom(P):
        def forward(self, x): w = torch.exp(x[:, 0] - 0.5); return torch.stack([1 / (1 + w), w / (1 + w)], 1)
    class Mod(P):
        def forward(self, x): return x[:, 1:]
    def draw(g, n, mu, easy_high):
        x = g.normal(mu, 1, n); conc = np.exp(1.2 * x) if easy_high else np.exp(-1.2 * x); y = g.integers(0, 3, n); A = np.full((n, 3), 1.0); A[np.arange(n), y] += 3 * conc
        Pr = np.array([g.dirichlet(a) for a in A]); y = np.array([g.choice(3, p=p) for p in Pr]); return torch.tensor(np.c_[x, np.log(Pr + 1e-30)], dtype=torch.float32), torch.tensor(y)
    try:
        p = WeightedPredictor(LAC(), Mod(), alpha=0.1, image_encoder=Enc(), domain_classifier=Dom()); print(f"  alpha passed to the constructor is 0.1; predictor.alpha after construction: {p.alpha}")
    except Exception as e: print("  constructor:", type(e).__name__, e)
    for easy_high, label in ((True, "model more accurate where the target has more mass"), (False, "model less accurate where the target has more mass")):
        ct, st, cr, sr, cu = [], [], [], [], []
        for r in range(R):
            g = np.random.default_rng(r); torch.manual_seed(r); xc, yc = draw(g, 1000, 0, easy_high); xt, yt = draw(g, 2000, 1, easy_high)
            p = WeightedPredictor(LAC(), Mod(), alpha=0.1, image_encoder=Enc(), domain_classifier=Dom()); p.calibrate(DataLoader(TensorDataset(xc, yc), batch_size=1000), alpha=0.1)
            res = p.evaluate(DataLoader(TensorDataset(xt, yt), batch_size=500)); ct.append(res["coverage_rate"]); st.append(res["average_size"])
            # reference: Tibshirani et al. weighted quantile, each weight kept with its own score
            sc = LAC()(xc[:, 1:], yc).numpy(); wc = np.exp(xc[:, 0].numpy() - 0.5); o = np.argsort(sc); sc, wc = sc[o], wc[o]; cw = np.cumsum(wc); St = LAC()(xt[:, 1:]).numpy(); wt = np.exp(xt[:, 0].numpy() - 0.5)
            k = (cw[None, :] / (cw[-1] + wt)[:, None] >= 0.9).argmax(1); ok = cw[-1] / (cw[-1] + wt) >= 0.9; q = np.where(ok, sc[k], np.inf); S = St <= q[:, None]; cr.append(S[np.arange(2000), yt.numpy()].mean()); sr.append(S.sum(1).mean())
            q0 = np.sort(sc)[math.ceil(1001 * 0.9) - 1]; cu.append((St <= q0)[np.arange(2000), yt.numpy()].mean())
        print(f"  {label}: TorchCP coverage {np.mean(ct):.4f}, set size {np.mean(st):.3f} | reference weighted method {np.mean(cr):.4f}, size {np.mean(sr):.3f} | no weighting {np.mean(cu):.4f}", flush=True)
if part in ("all", "cd"):
    from torchcp.classification.score import LAC, APS
    from torchcp.classification.predictor import SplitPredictor
    from torchcp.classification.loss import CDLoss
    print("(b) CDLoss (C-Adapter): which way does minimising it move the scores? Linear layer on a cross-entropy-trained start, 300 Adam steps, epsilon 0.5; split conformal set size on fresh data at 90%")
    def data(g, n): y = g.integers(0, 6, n); x = g.normal(size=(n, 6)) * 1.3; x[np.arange(n), y] += 1.5; return torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    for sname, mk in (("LAC", lambda: LAC()), ("APS", lambda: APS(randomized=False))):
        out = {"start": [], "minimise CDLoss": [], "minimise -CDLoss": []}; gap = {k: [] for k in out}
        for r in range(max(5, R // 4)):
            g = np.random.default_rng(r); torch.manual_seed(r); xtr, ytr = data(g, 2000); xc, yc = data(g, 1000); xt, yt = data(g, 3000)
            def size(lin):
                with torch.no_grad():
                    p = SplitPredictor(mk(), alpha=0.1); p.calculate_threshold(lin(xc), yc); S = p.predict_with_logits(lin(xt)); s = mk()(lin(xt)); sy = s[torch.arange(3000), yt]
                    return S.float().sum(1).mean().item(), ((s.sum(1) - sy) / 5 - sy).mean().item()
            base = torch.nn.Linear(6, 6); opt = torch.optim.Adam(base.parameters(), lr=0.05)
            for _ in range(200): opt.zero_grad(); torch.nn.functional.cross_entropy(base(xtr), ytr).backward(); opt.step()
            a, b = size(base); out["start"].append(a); gap["start"].append(b)
            for key, sign in (("minimise CDLoss", 1.0), ("minimise -CDLoss", -1.0)):
                lin = torch.nn.Linear(6, 6); lin.load_state_dict(base.state_dict()); opt = torch.optim.Adam(lin.parameters(), lr=0.02); L = CDLoss(SplitPredictor(mk(), alpha=0.1), epsilon=0.5)
                for _ in range(300): opt.zero_grad(); (sign * L(lin(xtr), ytr)).backward(); opt.step()
                a, b = size(lin); out[key].append(a); gap[key].append(b)
        print(f"  {sname}: " + "; ".join(f"{k}: set size {np.mean(v):.2f}, wrong-label score minus true-label score {np.mean(gap[k]):+.3f}" for k, v in out.items()), flush=True)
if part in ("all", "m"):
    from torchcp.classification.utils.metrics import Metrics
    from torchcp.llm.utils.metrics import Metrics as LM
    M = Metrics(); g = np.random.default_rng(0); n, K = 4000, 7; S = torch.tensor(g.random((n, K)) < 0.45).int(); y = torch.tensor(g.integers(0, K, n)); Sn, yn = S.numpy().astype(bool), y.numpy(); cov = Sn[np.arange(n), yn]
    cls = np.array([cov[yn == k].mean() for k in range(K)]); sz = Sn.sum(1)
    ref = {"coverage_rate": cov.mean(), "average_size": sz.mean(), "CovGap": 100 * np.abs(cls - 0.9).mean(), "VioClasses": (cls < 0.9).sum(),
           "SSCV": max(abs(0.9 - cov[(sz >= a) & (sz <= b)].mean()) for a, b in [[0, 1], [2, 3], [4, 10]] if ((sz >= a) & (sz <= b)).any()), "singleton_hit_ratio": ((sz == 1) & cov).mean()}
    got = {"coverage_rate": M("coverage_rate")(S, y), "average_size": M("average_size")(S), "CovGap": M("CovGap")(S, y, 0.1, K), "VioClasses": M("VioClasses")(S, y, 0.1, K), "SSCV": M("SSCV")(S, y, 0.1), "singleton_hit_ratio": M("singleton_hit_ratio")(S, y)}
    print("(c) classification metrics against an independent calculation"); [print(f"  {k:20s} TorchCP {float(got[k]):.6f}  independent {float(ref[k]):.6f}  {'same' if abs(float(got[k]) - float(ref[k])) < 1e-5 else 'DIFFERENT'}") for k in ref]
    print(f"  coverage_rate(macro) TorchCP {M('coverage_rate')(S, y, 'macro', K):.6f}  independent {cls.mean():.6f}")
    from torchcp.regression.utils.metrics import Metrics as RM
    iv = torch.tensor(np.sort(g.normal(size=(n, 1, 2)), -1), dtype=torch.float32); yy = torch.tensor(g.normal(size=n), dtype=torch.float32); a = RM()("coverage_rate")(iv, yy); b = ((yy >= iv[:, 0, 0]) & (yy <= iv[:, 0, 1])).float().mean().item()
    print(f"  regression coverage_rate TorchCP {a:.6f} independent {b:.6f}; average_size TorchCP {RM()('average_size')(iv):.6f} independent {(iv[:, 0, 1] - iv[:, 0, 0]).mean().item():.6f}")
    print("(c2) LLM metric SSCL (size-stratified loss): 300 examples, 8 generations")
    Sl = torch.tensor(np.cumsum(g.random((300, 8)) < 0.4, 1) < 2).float(); Ll = torch.tensor(g.random((300, 8)) < 0.3).float()
    try: print(f"  SSCL returns {LM()('SSCL')(Sl, Ll):.4f}; it sums set membership over examples (dim 0), giving {tuple(torch.sum(Sl, 0).shape)} 'sizes' for 300 examples")
    except Exception as e: print(f"  SSCL: {type(e).__name__}: {' '.join(str(e).split())[:160]}")
if part in ("all", "cpd"):
    from torchcp.regression.predictor import ConformalPredictiveDistribution, SplitPredictor as RS
    from torchcp.regression.score import Sign, R2CCP
    from torchcp.regression.utils.metrics import Metrics as RM
    class Mean(P):
        def forward(self, x): return x[:, :1]
    print("(d) ConformalPredictiveDistribution: P(y <= its tau-quantile) for tau = 0.1, 0.5, 0.9 (should be about tau)")
    hit = {0.1: [], 0.5: [], 0.9: []}
    for r in range(R):
        g = np.random.default_rng(r); mk = lambda n: (lambda x: (x, x[:, 0] + torch.tensor(g.standard_t(4, n), dtype=torch.float32)))(torch.tensor(g.normal(size=(n, 2)), dtype=torch.float32)); xc, yc = mk(199); xt, yt = mk(1000)
        c = ConformalPredictiveDistribution(Sign(), Mean()); c.calibrate(DataLoader(TensorDataset(xc, yc), batch_size=200)); V = torch.sort(c.predict(xt), 1)[0]
        for t in hit: hit[t].append((yt <= V[:, int(round(t * 200)) - 1]).float().mean().item())
    print("  " + "; ".join(f"tau {t}: {np.mean(v):.4f}" for t, v in hit.items()) + f"; output shape for 1,000 rows and 199 calibration rows: {tuple(V.shape)}, values returned unsorted")
    print("(e) R2CCP with SplitPredictor, trained (bimodal data, 40 bins, 60 epochs), 90% target")
    cov, sz = [], []
    for r in range(min(R, 8)):
        g = np.random.default_rng(r); torch.manual_seed(r)
        def mk(n): x = torch.tensor(g.normal(size=(n, 2)), dtype=torch.float32); s = torch.tensor(g.choice([-1.0, 1.0], n), dtype=torch.float32); return x, s * (1.5 + x[:, 0].abs()) + 0.3 * torch.tensor(g.normal(size=n), dtype=torch.float32)
        xn, yn = mk(1500); xc, yc = mk(800); xt, yt = mk(2000); mid = torch.linspace(float(yn.min()) - 0.5, float(yn.max()) + 0.5, 40)
        p = RS(R2CCP(mid), alpha=0.1); p.train(DataLoader(TensorDataset(xn, yn), batch_size=128, shuffle=True), epochs=60, lr=1e-3, verbose=False); p.calibrate(DataLoader(TensorDataset(xc, yc), batch_size=800)); iv = p.predict(xt)
        cov.append(RM()("coverage_rate")(iv, yt)); sz.append(RM()("average_size")(iv))
    print(f"  coverage {np.mean(cov):.4f} (se {np.std(cov) / np.sqrt(len(cov)):.4f}), mean total length {np.mean(sz):.3f}")
if part in ("all", "llm"):
    from torchcp.llm.predictor import ConformalLM
    from torchcp.llm.utils import loss as L
    from torchcp.llm.utils.metrics import Metrics as LM
    print("(f) ConformalLM risk control (no language model: synthetic scores). Risk = no correct answer in the set. Share of calibration draws whose chosen setting has true risk above epsilon; should be at most alpha = 0.05.")
    def draw(g, n, G=8): s = torch.tensor(g.random((n, G)), dtype=torch.float32); lab = torch.tensor(g.random((n, G)) < 0.35 * s.numpy() ** 2 * 3, dtype=torch.float32); sim = torch.tensor(g.random((n, G, G)), dtype=torch.float32); return s, sim, lab
    for scaling in ("none", "platt"):
        for rej in (False, True):
            try:
                g = np.random.default_rng(0); torch.manual_seed(0); eps = [0.3, 0.4]; lm = ConformalLM(epsilons=eps, scaling_type=scaling, set_score_function_name="geo", rejection=rej, alpha=0.05)
                s, sim, lab = draw(g, 400); lm.scaling(s, lab); s, sim, lab = draw(g, 150); lm.tuning(s, sim, lab); S, SIM, LAB = draw(g, 20000); viol = np.zeros(2); found = np.zeros(2); risks = [[], []]; cache = {}
                for r in range(R * 4):
                    s, sim, lab = draw(g, 300); lm.calibrate_configs(s, sim, lab)
                    for j in range(2):
                        cfg = lm.best_valid_configs[j]
                        if torch.isnan(cfg).any(): continue
                        key = tuple(cfg.tolist())
                        if key not in cache: cache[key] = LM()("average_set_loss")(lm.predict_with_config(cfg, S, SIM), L.set_losses_from_labels(LAB))
                        found[j] += 1; viol[j] += cache[key] > eps[j]; risks[j].append(cache[key])
                print(f"  scaling {scaling}, rejection {rej}: " + "; ".join(f"epsilon {eps[j]}: above epsilon in {viol[j] / (R * 4):.3f} of draws (setting found in {found[j] / (R * 4):.2f}, mean true risk {np.mean(risks[j]) if risks[j] else float('nan'):.3f})" for j in range(2)) + f"; candidate settings {len(lm.candidate_configs)}", flush=True)
            except Exception as e: print(f"  scaling {scaling}, rejection {rej}: {type(e).__name__}: {' '.join(str(e).split())[:200]}", flush=True)
