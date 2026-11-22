"""The same sequences through an unclipped ACI (pyconformal, and a 12-line loop written from the paper)."""
import numpy as np, math, pyconformal as pc
T, n0, gamma, alpha = 3000, 200, 0.05, 0.1
bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
rng = np.random.default_rng(0)
cases = {
    "i.i.d. noise": rng.normal(size=n0 + T),
    "noise scale x10 halfway": np.r_[rng.normal(size=n0 + T // 2), 10 * rng.normal(size=T - T // 2)],
    "errors growing 1% per step": rng.normal(size=n0 + T) * 1.01 ** np.arange(n0 + T),
    "errors growing 5% per step": rng.normal(size=n0 + T) * 1.05 ** np.arange(n0 + T),
    "every outcome larger than all before": np.arange(1.0, n0 + T + 1) ** 2,
}
def paper_loop(s, n0, clip):
    a, miss, past = alpha, [], list(s[:n0])
    for t in range(n0, len(s)):
        n = len(past)
        if a <= 0: q = math.inf
        elif a >= 1: q = -math.inf
        else:
            k = math.ceil((n + 1) * (1 - a)); q = math.inf if k > n else np.partition(past, k - 1)[k - 1]
        err = s[t] > q; miss.append(err); a = a + gamma * (alpha - err)
        if clip: a = min(max(a, 0.0), 1.0)
        past.append(s[t])
    return np.mean(miss)
print(f"bound on |miss rate - {alpha}|: {bound:.4f}")
for name, y in cases.items():
    s = np.abs(y)
    r = pc.AdaptiveConformalInference(alpha=alpha, gamma=gamma).calibrate(scores=s[:n0]).run(scores=s[n0:])
    print(f"  {name:40s} pyconformal {r.errors.mean():.4f}   paper loop {paper_loop(s, n0, False):.4f}   paper loop with alpha clipped to [0, 1] {paper_loop(s, n0, True):.4f}")
