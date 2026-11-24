import sys, types, numpy as np, re
src = open("multiQT/example.py").read(); a = src.index("def isotonic_regression"); b = src.index('if __name__ == "__main__"')
ns = {"np": np}; exec("from sklearn.isotonic import IsotonicRegression\n" + src[a:b], ns); theirs = ns["multiQT"]
from pyconformal.multiqt import MultiQT
rng = np.random.default_rng(0); worst = 0
for case in range(12):
    T = 400; levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] if case % 2 else [0.1, 0.5, 0.9]
    Y = np.cumsum(rng.normal(size=T)) * (0.2 if case % 3 else 2.0) + 5 * np.sin(np.arange(T) / 30)
    base = np.stack([np.r_[0, Y[:-1]] + q for q in np.quantile(rng.normal(size=500), levels)]) * (1 if case % 4 else 0)
    d = [0, 0, 3, 1][case % 4]; lr = ["adaptive+", 0.3][case // 6]
    delay = [[] for _ in range(d)] + [[t] for t in range(T - d)]
    q1 = theirs(Y, levels, base_forecasts=base.copy(), lr=lr, delay=delay)
    q2 = MultiQT(levels, lr="adaptive" if lr == "adaptive+" else lr).run(base.T, Y, delay=d).quantiles[:, 0, :].T
    worst = max(worst, np.abs(q1 - q2).max()); print(case, "delay", d, "lr", lr, "levels", len(levels), "zero base", not (case % 4), "diff", float(np.abs(q1 - q2).max()), "first differing step", int(np.argmax(np.abs(q1 - q2).max(0) > 1e-9)))
print("largest absolute difference from the authors' code over 12 cases:", worst)
