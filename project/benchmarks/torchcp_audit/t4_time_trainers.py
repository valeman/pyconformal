"""TorchCP 1.2.1: ACIPredictor.evaluate and EnsemblePredictor.evaluate end to end on dependent data; trainers followed by split calibration."""
import sys, io, contextlib, warnings, numpy as np, torch
warnings.filterwarnings("ignore")
from torch.utils.data import TensorDataset, DataLoader
R = int(sys.argv[1]); part = sys.argv[2]
def series(kind, g, n):
    e = g.normal(size=n); t = np.arange(n)
    if kind == "AR(1), stable": s = np.ones(n); d = 0
    elif kind == "AR(1), noise x4 from the middle": s = np.where(t < n * 0.6, 1, 4); d = 0
    elif kind == "AR(1), noise grows 0.5% a step": s = 1.005 ** t; d = 0
    y = np.zeros(n)
    for i in range(1, n): y[i] = 0.6 * y[i - 1] + s[i] * e[i]
    X = np.c_[np.r_[0, y[:-1]], np.r_[0, 0, y[:-2]]]; return torch.tensor(X[2:], dtype=torch.float32), torch.tensor(y[2:], dtype=torch.float32)
kinds = ["AR(1), stable", "AR(1), noise x4 from the middle", "AR(1), noise grows 0.5% a step"]
if part == "aci":
    from torchcp.regression.score import ABS
    from torchcp.regression.predictor import ACIPredictor, EnsemblePredictor
    print("ACIPredictor.train + .evaluate (lookback 200, retrain every 25 steps, 30 epochs), linear model, 400 training steps, 1,000 test steps, 90% target")
    for kind in kinds:
        for gamma in (0.005, 0.05):
            cov = []
            for r in range(R):
                g = np.random.default_rng(r); torch.manual_seed(r); X, y = series(kind, g, 1402); tr = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=100); te = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=1)
                p = ACIPredictor(ABS(), torch.nn.Linear(2, 1), gamma=gamma, alpha=0.1)
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    p.train(tr, epochs=30, verbose=False); orig = p.score_function.train; p.score_function.train = lambda dl, **kw: orig(dl, **{**kw, "epochs": 30}); res = p.evaluate(te, lookback=200, retrain_gap=25, update_alpha_gap=1)
                cov.append(res["coverage_rate"])
            print(f"  {kind:32s} gamma {gamma}: miss rate {1 - np.mean(cov):.4f} (worst run {1 - np.min(cov):.4f}); Gibbs-Candes bound {0.1 + (0.9 + gamma) / (1000 * gamma):.4f}", flush=True)
    print("EnsemblePredictor.train + .evaluate (EnbPI, 10 models, updates after each batch of 1), same series")
    for kind in kinds:
        cov = []
        for r in range(R):
            g = np.random.default_rng(r); torch.manual_seed(r); X, y = series(kind, g, 1402); tr = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=100); te = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=1)
            p = EnsemblePredictor(ABS(), torch.nn.Linear(2, 1), alpha=0.1); p.train(tr, ensemble_num=10, subset_num=300, epochs=30, verbose=False); cov.append(p.evaluate(te, verbose=False)["coverage_rate"])
        print(f"  {kind:32s} coverage {np.mean(cov):.4f} (lowest run {np.min(cov):.4f})   [EnbPI has no finite-sample guarantee under drift]", flush=True)
if part == "tr":
    from torchcp.classification.score import APS, LAC
    from torchcp.classification.predictor import SplitPredictor
    from torchcp.classification import trainer as T
    print("Trainers, then split conformal on fresh data (APS, 90%). 6 classes, 3,000 training rows. Coverage must stay on target whatever the trainer does; set size is the trainer's claim.")
    def data(g, n): y = g.integers(0, 6, n); x = g.normal(size=(n, 6)) * 1.3; x[np.arange(n), y] += 1.5; return torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    def net(): return torch.nn.Sequential(torch.nn.Linear(6, 32), torch.nn.ReLU(), torch.nn.Linear(32, 6))
    def pre(xtr, ytr):
        m = net(); o = torch.optim.Adam(m.parameters(), lr=0.01)
        for _ in range(150): o.zero_grad(); torch.nn.functional.cross_entropy(m(xtr), ytr).backward(); o.step()
        return m
    specs = {"cross-entropy only": lambda m, tl, vl: m,
             "TSTrainer": lambda m, tl, vl: (lambda t: (t.train(vl, num_epochs=50), t.model)[1])(T.TSTrainer(m, init_temperature=1.0, verbose=False)),
             "ConfTSTrainer": lambda m, tl, vl: (lambda t: (t.train(vl, num_epochs=50), t.model)[1])(T.ConfTSTrainer(m, alpha=0.1, init_temperature=1.0, verbose=False)),
             "ConfTrTrainer": lambda m, tl, vl: (lambda t: (t.train(tl, num_epochs=10), t.model)[1])(T.ConfTrTrainer(alpha=0.1, model=m, verbose=False)),
             "SCPOTrainer": lambda m, tl, vl: (lambda t: (t.train(tl, num_epochs=10), t.model)[1])(T.SCPOTrainer(m, alpha=0.1, verbose=False)),
             "UncertaintyAwareTrainer": lambda m, tl, vl: (lambda t: (t.train(tl, num_epochs=10), t.model)[1])(T.UncertaintyAwareTrainer(m, weight=0.2, verbose=False)),
             "OrdinalTrainer": lambda m, tl, vl: (lambda t: (t.train(tl, num_epochs=10), t.model)[1])(T.OrdinalTrainer(m, ordinal_config={"phi": "abs", "varphi": "abs"}, verbose=False))}
    for name, fn in specs.items():
        cov, sz, acc, err = [], [], [], None
        for r in range(R):
            g = np.random.default_rng(r); torch.manual_seed(r); xtr, ytr = data(g, 3000); xv, yv = data(g, 1000); xc, yc = data(g, 1000); xt, yt = data(g, 3000)
            tl = DataLoader(TensorDataset(xtr, ytr), batch_size=200, shuffle=True); vl = DataLoader(TensorDataset(xv, yv), batch_size=200)
            try:
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()): m = fn(pre(xtr, ytr), tl, vl)
                m.eval()
                with torch.no_grad(): Lc, Lt = m(xc), m(xt)
                sc = LAC(score_type="identity") if name == "SCPOTrainer" else APS()
                p = SplitPredictor(sc, alpha=0.1); p.calculate_threshold(Lc, yc); S = p.predict_with_logits(Lt)
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:170]}"; break
            cov.append(S[torch.arange(3000), yt].float().mean().item()); sz.append(S.float().sum(1).mean().item()); acc.append((Lt.argmax(1) == yt).float().mean().item())
        print(f"  {name:24s} " + (f"ERROR {err}" if err else f"coverage {np.mean(cov):.4f}, set size {np.mean(sz):.2f}, accuracy {np.mean(acc):.3f}" + (" (scored with LAC on the surrogate outputs)" if name == "SCPOTrainer" else "")), flush=True)
