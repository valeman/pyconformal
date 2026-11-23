"""PUNCC 0.9.3: box-wise conformal object detection with a detector whose boxes are a little too small.
Prints the share of true boxes that lie inside the 'outer' conformal box (target at least 0.90)."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from deel.puncc.object_detection import SplitBoxWise
from deel.puncc.api.prediction import IdPredictor
from deel.puncc.metrics import object_detection_mean_coverage

g = np.random.default_rng(0); n = 1300
centre = g.uniform(100, 400, (n, 2)); size = g.uniform(40, 120, (n, 2))
true = np.column_stack([centre - size / 2, centre + size / 2])            # (x_min, y_min, x_max, y_max)
shrink = g.gamma(2.0, 4.0, (n, 4))                                         # every side predicted a few pixels inside the truth
pred = true + np.column_stack([shrink[:, 0], shrink[:, 1], -shrink[:, 2], -shrink[:, 3]])

for method in ("additive", "multiplicative"):
    od = SplitBoxWise(IdPredictor(), train=False, method=method)
    od.fit(X_calib=pred[:300], y_calib=true[:300])
    _, inner, outer = od.predict(pred[300:], alpha=0.1)
    print(f"{method:14s}: coverage {object_detection_mean_coverage(np.asarray(outer), true[300:]):.3f} (PUNCC's own metric)")
