import warnings, numpy as np; warnings.filterwarnings("ignore")
from crepes import ConformalPredictiveSystem, ConformalRegressor
g = np.random.default_rng(0); n, T = 60, 80; bins = g.integers(0, 2, n + T); y = g.normal(size=n + T)
on = ConformalPredictiveSystem().fit(y[:n], bins=bins[:n]).predict_int_online(np.zeros(T), y[n:], bins=bins[n:], confidence=0.9)
bad = 0; first = None
for t in range(T):
    b = ConformalPredictiveSystem().fit(y[:n + t], bins=bins[:n + t]).predict_int(np.zeros(1), bins=bins[n + t:n + t + 1], confidence=0.9)[0]
    same = np.allclose(on[t], b, equal_nan=True)
    if not same:
        bad += 1
        if first is None: first = (t, on[t], b, int((bins[:n + t] == bins[n + t]).sum()))
print(f"Mondrian predictive system, online against batch refit: {bad} of {T} steps differ; first: step, online, batch, calibration size in that category = {first}")
# which one is right? exact ranks by hand for the first differing step
t, o, b, m = first; s = np.sort(y[:n + t][bins[:n + t] == bins[n + t]]); import math
kl = math.floor((m + 1) * 0.05); kh = math.ceil((m + 1) * 0.95)
print(f"  by hand: category has {m} residuals; lower rank {kl}, upper rank {kh}: [{s[kl-1] if kl >= 1 else -np.inf:.4f}, {s[kh-1] if kh <= m else np.inf:.4f}]")
cov_on = ((on[:, 0] <= y[n:]) & (y[n:] <= on[:, 1])).mean()
# coverage of the online Mondrian predictive system over many runs
cov = []; covb = []; wid = []; widb = []
for r in range(400):
    g = np.random.default_rng(r); bins = g.integers(0, 2, n + T); y = g.normal(size=n + T) * np.array([1.0, 3.0])[bins]
    on = ConformalPredictiveSystem().fit(y[:n], bins=bins[:n]).predict_int_online(np.zeros(T), y[n:], bins=bins[n:], confidence=0.9)
    cov.append(((on[:, 0] <= y[n:]) & (y[n:] <= on[:, 1])).mean()); fin = np.isfinite(on).all(1); wid.append(np.mean((on[:, 1] - on[:, 0])[fin]))
    for c in (0, 1): pass
print(f"  online Mondrian predictive system, two categories with different noise: coverage {np.mean(cov):.4f} (se {np.std(cov)/20:.4f}) at 90%")
# per-category coverage, which is what Mondrian promises
cc = {0: [], 1: []}
for r in range(400):
    g = np.random.default_rng(r); bins = g.integers(0, 2, n + T); y = g.normal(size=n + T) * np.array([1.0, 3.0])[bins]
    on = ConformalPredictiveSystem().fit(y[:n], bins=bins[:n]).predict_int_online(np.zeros(T), y[n:], bins=bins[n:], confidence=0.9); hit = (on[:, 0] <= y[n:]) & (y[n:] <= on[:, 1])
    for c in (0, 1): cc[c].append(hit[bins[n:] == c].mean())
print(f"  per category: low-noise {np.mean(cc[0]):.4f}, high-noise {np.mean(cc[1]):.4f}")
