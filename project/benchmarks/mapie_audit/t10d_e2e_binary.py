"""End to end through the public class, with a 0/1 loss: one true label per row, so recall for a row is 0 or 1.
The score of the true label is uniform on (0, 1), so the true risk at threshold lam is exactly lam (the label is kept when score > lam)."""
import sys, numpy as np, warnings
from mapie.risk_control import MultiLabelClassificationController
warnings.simplefilter("ignore")
R = int(sys.argv[1])
print("MultiLabelClassificationController(risk='recall', method='rcps', rcps_bound=None): share of runs whose TRUE recall is below the target")
for target, conf, n in ((0.8, 0.9, 100), (0.9, 0.9, 100), (0.8, 0.95, 100), (0.8, 0.9, 200)):
    bad = 0; lams = []
    for r in range(R):
        rng = np.random.default_rng(32452843 * r + n)
        P = rng.random((n, 1)); Y = np.ones((n, 1), dtype=int)
        c = MultiLabelClassificationController(lambda X: np.asarray(X), risk="recall", method="rcps", target_level=target, confidence_level=conf)
        c.calibrate(P, Y)
        lam = float(np.ravel(c.best_predict_param)[0]); lams.append(lam)
        bad += lam > (1 - target) + 1e-12          # true risk = lam
    d = 1 - conf; se = np.sqrt(d * (1 - d) / R)
    print(f"  recall >= {target}, confidence {conf}, {n} calibration rows: {bad/R:.3f} of {R} runs (allowed {d:.2f})  {'VIOLATED' if bad/R > d + 3*se else 'within noise of the limit' if bad/R > d else 'ok'}   median threshold {np.median(lams):.2f}", flush=True)
