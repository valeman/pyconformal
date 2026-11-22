"""MAPIE issue #1021: APS prediction sets with include_last_label=False under-cover.
MAPIE 1.6.0, 200 repetitions, 500 calibration and 1,000 test points, 5 classes, 90% target:
include_last_label=True 0.950, False 0.811, "randomized" 0.901."""
import warnings, numpy as np, mapie, inspect
warnings.filterwarnings("ignore")
print("MAPIE", mapie.__version__)
from mapie.classification import SplitConformalClassifier
from mapie.regression import ConformalizedQuantileRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class Fixed(BaseEstimator, ClassifierMixin):
    def __init__(self, k=5): self.k = k
    def fit(self, X, y): self.classes_ = np.arange(self.k); return self
    def predict_proba(self, X): return X[:, :self.k]
    def predict(self, X): return X[:, :self.k].argmax(1)
    def __sklearn_is_fitted__(self): return True

# 1. APS with include_last_label=False (issue #1021)
rng = np.random.default_rng(0); K = 5; cov = {}
for rep in range(200):
    P = rng.dirichlet(np.ones(K) * 0.7, 1500); y = np.array([rng.choice(K, p=p) for p in P])
    for ill in (True, False, "randomized"):
        try:
            m = SplitConformalClassifier(Fixed(K).fit(P, y), confidence_level=0.9, conformity_score="aps", prefit=True)
            m.conformalize(P[:500], y[:500])
            S = m.predict_set(P[500:], conformity_score_params={"include_last_label": ill})[1][:, :, 0]
            cov.setdefault(str(ill), []).append(S[np.arange(1000), y[500:]].mean())
        except Exception as e:
            cov[str(ill)] = repr(e)[:120]; 
for k, v in cov.items():
    print("APS include_last_label=%s:" % k, v if isinstance(v, str) else "coverage %.4f +- %.4f (target 0.90)" % (np.mean(v), 2 * np.std(v) / np.sqrt(len(v))))
