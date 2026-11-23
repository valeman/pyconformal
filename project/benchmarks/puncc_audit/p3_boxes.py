"""PUNCC 0.9.3 SplitBoxWise. Boxes are (x_min, y_min, x_max, y_max). The score is prediction minus truth for all four coordinates,
and the outer box is [x_min - q0, y_min - q1, x_max + q2, y_max + q3]. For the two 'max' coordinates the score needed is truth minus prediction.
With errors symmetric around zero the two have the same distribution, so the fault is invisible; with a detector whose boxes are
systematically too small or too large it is not."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from deel.puncc.object_detection import SplitBoxWise
from deel.puncc.api.prediction import IdPredictor
from deel.puncc.metrics import object_detection_mean_coverage
R = 60
def boxes(g, n, kind):
    c = g.uniform(100, 400, (n, 2)); wh = g.uniform(40, 120, (n, 2)); T = np.column_stack([c - wh / 2, c + wh / 2])
    if kind == "symmetric errors": E = g.normal(0, 6, (n, 4))
    elif kind == "predicted boxes too small": s = g.gamma(2.0, 4.0, (n, 4)); E = np.column_stack([s[:, 0], s[:, 1], -s[:, 2], -s[:, 3]])       # every side pulled inwards
    elif kind == "predicted boxes too large": s = g.gamma(2.0, 4.0, (n, 4)); E = np.column_stack([-s[:, 0], -s[:, 1], s[:, 2], s[:, 3]])
    elif kind == "predicted boxes shifted right and down": s = g.gamma(2.0, 4.0, (n, 4)); E = s
    return T, T + E
def contains(outer, T): return (outer[:, 0] <= T[:, 0]) & (outer[:, 1] <= T[:, 1]) & (outer[:, 2] >= T[:, 2]) & (outer[:, 3] >= T[:, 3])
def by_definition(Pc, Tc, Pt, alpha):     # box-wise conformal as defined in the method: scores point outwards on every side, Bonferroni over the four sides
    S = np.column_stack([Pc[:, 0] - Tc[:, 0], Pc[:, 1] - Tc[:, 1], Tc[:, 2] - Pc[:, 2], Tc[:, 3] - Pc[:, 3]]); n = len(S); k = int(np.ceil((n + 1) * (1 - alpha / 4)))
    q = np.sort(S, axis=0)[k - 1]; return np.column_stack([Pt[:, 0] - q[0], Pt[:, 1] - q[1], Pt[:, 2] + q[2], Pt[:, 3] + q[3]])
print("Share of test boxes whose true box lies inside the outer box (target at least 0.90). 300 calibration boxes, 1,000 test boxes.")
for kind in ("symmetric errors", "predicted boxes too small", "predicted boxes too large", "predicted boxes shifted right and down"):
    for method in ("additive", "multiplicative"):
        cov = []; met = []; ref = []; side = []
        for r in range(R):
            g = np.random.default_rng(r); T, P = boxes(g, 1300, kind)
            m = SplitBoxWise(IdPredictor(), train=False, method=method); m.fit(X_calib=P[:300], y_calib=T[:300]); _, inner, outer = m.predict(P[300:], alpha=0.1); outer = np.asarray(outer)
            cov.append(contains(outer, T[300:]).mean()); met.append(object_detection_mean_coverage(outer, T[300:])); ref.append(contains(by_definition(P[:300], T[:300], P[300:], 0.1), T[300:]).mean())
            side.append([(outer[:, 0] <= T[300:, 0]).mean(), (outer[:, 1] <= T[300:, 1]).mean(), (outer[:, 2] >= T[300:, 2]).mean(), (outer[:, 3] >= T[300:, 3]).mean()])
        s = np.mean(side, axis=0)
        print(f"  {kind:40s} {method:14s}: PUNCC {np.mean(cov):.4f} (its own metric: {np.mean(met):.4f}); per side x_min {s[0]:.3f}, y_min {s[1]:.3f}, x_max {s[2]:.3f}, y_max {s[3]:.3f}" + (f"; by definition (additive) {np.mean(ref):.4f}" if method == "additive" else ""), flush=True)
