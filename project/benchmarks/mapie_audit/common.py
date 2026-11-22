import warnings, logging, numpy as np
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class FixedClf(BaseEstimator, ClassifierMixin):
    """A 'model' whose probabilities are the features themselves."""
    def __init__(self, k=5): self.k = k
    def fit(self, X, y=None): self.classes_ = np.arange(self.k); return self
    def predict_proba(self, X): return np.asarray(X)[:, :self.k]
    def predict(self, X): return np.asarray(X)[:, :self.k].argmax(1)
    def __sklearn_is_fitted__(self): return True

class Const(BaseEstimator, RegressorMixin):
    """Predicts a constant (or column 0 of X times slope plus the constant)."""
    def __init__(self, c=0.0, slope=0.0): self.c = c; self.slope = slope
    def fit(self, X, y=None): self.fitted_ = True; return self
    def predict(self, X): return self.c + self.slope * np.asarray(X)[:, 0]
    def __sklearn_is_fitted__(self): return True

def summary(v):
    v = np.asarray(v, float); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

def verdict(mean, se, target):
    return "UNDER" if mean < target - 3 * se else ("over" if mean > target + 0.03 else "ok")
