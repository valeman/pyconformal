"""SplitCAD with a fixed score (distance from the origin), so only the calibration rule is tested. 2,000 repetitions."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from deel.puncc.anomaly_detection import SplitCAD
from deel.puncc.api.prediction import BasePredictor
class Norm(BasePredictor):
    def __init__(self): super().__init__(model=None, is_trained=True)
    def fit(self, X, y=None, **k): pass
    def predict(self, X): return np.linalg.norm(np.asarray(X), axis=1)
R = 2000
for ncal in (19, 20, 99, 100, 500):
    for alpha in (0.1, 0.05):
        fa = []; err = None
        for r in range(R):
            g = np.random.default_rng(r); z = g.normal(size=(ncal + 400, 3))
            try: m = SplitCAD(Norm(), train=False); m.fit(z_calib=z[:ncal]); fa.append(np.mean(np.asarray(m.predict(z[ncal:], alpha=alpha))))
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"; break
        if err: print(f"  {ncal:3d} calibration points, alpha {alpha}: {err}"); continue
        import math; exact = 1 - math.ceil((1 - alpha) * (ncal + 1) - 1e-9) / (ncal + 1)
        m_, se = np.mean(fa), np.std(fa) / np.sqrt(R); print(f"  {ncal:3d} calibration points, alpha {alpha}: normal points flagged {m_:.4f} (se {se:.4f}); exact conformal rule gives {exact:.4f}  {'ABOVE alpha' if m_ > alpha + 3 * se else 'ok'}")
