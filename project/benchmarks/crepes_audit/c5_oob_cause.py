"""Cause of the out-of-bag + k-NN result: WrapRegressor.calibrate calls de.apply(X), which for k-NN estimators counts each training object as its own
nearest neighbour. The leave-one-out estimates that fit(oob=True) stored are only returned by de.apply() with no argument."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from crepes import WrapRegressor, ConformalRegressor
from crepes.extras import DifficultyEstimator
reps = int(sys.argv[1])
def data(g, n):
    X = g.uniform(-2, 2, (n, 5)); y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + g.normal(size=n) * (0.2 + np.abs(X[:, 2])); return X, y
for kind in ("residuals", "distance", "labels"):
    for k in (25, 5):
        a, b, same = [], [], []
        for r in range(reps):
            g = np.random.default_rng(1000 + r); X, y = data(g, 900); Xt, yt = data(g, 2000)
            w = WrapRegressor(RandomForestRegressor(n_estimators=100, oob_score=True, random_state=r)); w.fit(X, y); res = y - w.learner.oob_prediction_
            kw = {"residuals": dict(residuals=res), "distance": {}, "labels": dict(y=y)}[kind]
            de = DifficultyEstimator().fit(X=X, scaler=True, oob=True, k=k, **kw)
            iv = w.calibrate(X, y, de=de, oob=True).predict_int(Xt, confidence=0.9); a.append(((iv[:, 0] <= yt) & (yt <= iv[:, 1])).mean())
            cr = ConformalRegressor().fit(res, sigmas=de.apply()); iv2 = cr.predict_int(w.predict(Xt), sigmas=de.apply(Xt), confidence=0.9); b.append(((iv2[:, 0] <= yt) & (yt <= iv2[:, 1])).mean())
            same.append(np.allclose(de.apply(), de.apply(X)))
        print(f"k-NN {kind:9s} k={k:2d}: through WrapRegressor.calibrate(X, y, de=de, oob=True): {np.mean(a):.4f} (se {np.std(a)/np.sqrt(reps):.4f}); with the stored leave-one-out estimates de.apply(): {np.mean(b):.4f} (se {np.std(b)/np.sqrt(reps):.4f}); de.apply() equals de.apply(X_train): {all(same)}", flush=True)
