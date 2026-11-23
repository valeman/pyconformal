"""PUNCC 0.9.3 SplitCAD, used as its docstring shows (the predictor returns the anomaly score). Share of NORMAL points flagged must be at most alpha."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import IsolationForest
from deel.puncc.anomaly_detection import SplitCAD
from deel.puncc.api.prediction import BasePredictor
class ADPredictor(BasePredictor):
    def predict(self, X): return -self.model.score_samples(X)
R = 80
for ncal in (100, 500):
    for alpha in (0.1, 0.05, 0.01):
        fa = []; det = []; err = None
        for r in range(R):
            g = np.random.default_rng(r); z = g.normal(size=(500 + ncal + 1000, 3))
            try:
                m = SplitCAD(ADPredictor(IsolationForest(n_estimators=50, random_state=r)), train=True, random_state=r); m.fit(z_fit=z[:500], z_calib=z[500:500 + ncal])
                fa.append(np.mean(np.asarray(m.predict(z[500 + ncal:], alpha=alpha)))); det.append(np.mean(np.asarray(m.predict(g.normal(size=(300, 3)) * 3.5, alpha=alpha))))
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:170]}"; break
        if err: print(f"  {ncal} calibration points, alpha {alpha}: ERROR {err}"); continue
        m_, se = np.mean(fa), np.std(fa) / np.sqrt(len(fa)); print(f"  {ncal:3d} calibration points, alpha {alpha}: normal points flagged {m_:.4f} (se {se:.4f}) {'ABOVE alpha' if m_ > alpha + 3 * se else 'ok'}; true anomalies flagged {np.mean(det):.3f}")
