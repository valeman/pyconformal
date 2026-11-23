"""crepes 0.9.1: out-of-bag calibration with a k-nearest-neighbour difficulty estimator, written as in crepes' own notebook
docs/crepes_nb_wrap.ipynb. Prints coverage at a 90% level."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from crepes import WrapRegressor
from crepes.extras import DifficultyEstimator

def data(g, n):
    X = g.uniform(-2, 2, (n, 5))
    return X, np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + g.normal(size=n) * (0.2 + np.abs(X[:, 2]))

cov = []
for r in range(20):
    g = np.random.default_rng(r)
    X_train, y_train = data(g, 900)
    X_test, y_test = data(g, 2000)
    learner_full = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=r)
    rf = WrapRegressor(learner_full)
    rf.fit(X_train, y_train)
    residuals_oob = y_train - learner_full.oob_prediction_
    de_knn_res_oob = DifficultyEstimator()
    de_knn_res_oob.fit(X=X_train, residuals=residuals_oob, scaler=True, oob=True, k=5)
    rf.calibrate(X_train, y_train, de=de_knn_res_oob, oob=True)
    iv = rf.predict_int(X_test, confidence=0.9)
    cov.append(np.mean((iv[:, 0] <= y_test) & (y_test <= iv[:, 1])))
print(f"coverage {np.mean(cov):.3f} at a 90% level")
