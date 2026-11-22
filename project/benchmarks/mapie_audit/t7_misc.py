"""Open issues retested on MAPIE 1.6.0, and a few lifecycle checks."""
import warnings, numpy as np, mapie
from common import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from mapie.regression import SplitConformalRegressor, CrossConformalRegressor, TimeSeriesRegressor
from mapie.classification import SplitConformalClassifier
from mapie.utils import train_conformalize_test_split
print("MAPIE", mapie.__version__)
rng = np.random.default_rng(0); X = rng.normal(size=(1000, 3)); y = X[:, 0] + rng.normal(size=1000)

def attempt(name, f):
    try: print(f"{name}: {f()}")
    except Exception as e: print(f"{name}: ERROR {type(e).__name__}: {str(e)[:160]}")

# 1018: split sizes for 0.8 / 0.1 / 0.1
attempt("#1018 split sizes for 0.8/0.1/0.1 of 1,000 rows", lambda: [len(a) for a in train_conformalize_test_split(X, y, train_size=0.8, conformalize_size=0.1, test_size=0.1, random_state=0)[:3]])
attempt("      split sizes for 0.6/0.2/0.2 of 1,000 rows", lambda: [len(a) for a in train_conformalize_test_split(X, y, train_size=0.6, conformalize_size=0.2, test_size=0.2, random_state=0)[:3]])
# 710: fit_conformalize twice
def twice():
    m = CrossConformalRegressor(LinearRegression(), cv=5); m.fit_conformalize(X[:200], y[:200]); m.fit_conformalize(X[:200], y[:200]); return "works"
attempt("#710 CrossConformalRegressor.fit_conformalize called twice", twice)
# 542: LeaveOneGroupOut
def logo():
    g = np.repeat(np.arange(10), 20); m = CrossConformalRegressor(LinearRegression(), cv=LeaveOneGroupOut()); m.fit_conformalize(X[:200], y[:200], groups=g)
    return m.predict_interval(X[200:205])[1].shape
attempt("#542 CrossConformalRegressor with cv=LeaveOneGroupOut()", logo)
# 484: several confidence levels in time series
def ts_levels():
    from mapie.subsample import BlockBootstrap
    m = TimeSeriesRegressor(LinearRegression(), method="enbpi", cv=BlockBootstrap(n_resamplings=10, n_blocks=5, overlapping=False, random_state=0)).fit(X[:300], y[:300])
    return m.predict(X[300:305], confidence_level=[0.8, 0.9], ensemble=True, optimize_beta=True)[1].shape
attempt("#484 TimeSeriesRegressor, two confidence levels, optimize_beta=True", ts_levels)
# stale calibration: conformalize, then refit the underlying model
def stale():
    est = LinearRegression().fit(X[:300], y[:300])
    m = SplitConformalRegressor(est, confidence_level=0.9, prefit=True).conformalize(X[300:600], y[300:600])
    w0 = np.diff(m.predict_interval(X[600:601])[1][0, :, 0])[0]
    est.fit(X[:300], 100 * y[:300])        # the same object, now a different model
    iv = m.predict_interval(X[600:])[1][:, :, 0]
    cov = ((iv[:, 0] <= 100 * y[600:]) & (100 * y[600:] <= iv[:, 1])).mean()
    return f"no error or warning; interval width unchanged at {w0:.2f}; coverage of the refitted model's intervals {cov:.3f} at a 0.9 target"
attempt("Refit the model after conformalizing (prefit object reused)", stale)
# gamma passed to update() is ignored
def upd():
    from sklearn.model_selection import PredefinedSplit
    m = TimeSeriesRegressor(Const(0.0), method="aci", cv=PredefinedSplit([-1] + [0] * 99)).fit(np.zeros((100, 1)), y[:100])
    m.predict(np.zeros((1, 1)), confidence_level=0.9)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always"); m.update(np.zeros((1, 1)), np.array([1e9]), confidence_level=0.9, gamma=0.5)
    return f"alpha_t after a miss with update(gamma=0.5): {m.current_alpha}; warnings raised: {len(w)}"
attempt("ACI: update(gamma=...) on its own", upd)
# NaN in probabilities / inputs
def nanp():
    P = np.tile([0.6, 0.4], (50, 1)); m = SplitConformalClassifier(FixedClf(2).fit(P), prefit=True).conformalize(P, np.zeros(50, int))
    return m.predict_set(np.array([[np.nan, 1.0]]))[1][0, :, 0]
attempt("NaN probability at prediction", nanp)
def cols():
    P = np.tile([0.6, 0.4], (50, 1)); m = SplitConformalClassifier(FixedClf(2).fit(P), prefit=True).conformalize(P, np.zeros(50, int))
    m._mapie_classifier.estimator_.single_estimator_.k = 3 if hasattr(m, "_mapie_classifier") else None
    return "n/a"
