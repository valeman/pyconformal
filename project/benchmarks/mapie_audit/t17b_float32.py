import numpy as np, warnings; warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from mapie.regression import SplitConformalRegressor, CrossConformalRegressor, ConformalizedQuantileRegressor
from mapie.conformity_scores import AbsoluteConformityScore, GammaConformityScore
rng = np.random.default_rng(0); n = 1200; X = rng.normal(size=(n, 4)); y = 50 + X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=n)
def go(label, f):
    try: f(); print(f"{label:70s} runs")
    except Exception as e: print(f"{label:70s} ERROR {type(e).__name__}: {' '.join(str(e).split())[:230]}")
for est_name, est in (("LinearRegression", LinearRegression), ("RandomForest", lambda: RandomForestRegressor(n_estimators=20, random_state=0)), ("HistGradientBoosting", HistGradientBoostingRegressor)):
    for xt, yt_ in (("float64", "float64"), ("float32", "float64"), ("float64", "float32"), ("float32", "float32")):
        Xa, ya = X.astype(xt), y.astype(yt_)
        go(f"split, {est_name}, X {xt}, y {yt_}", lambda: SplitConformalRegressor(est(), confidence_level=0.9, prefit=False).fit(Xa[:500], ya[:500]).conformalize(Xa[500:900], ya[500:900]).predict_interval(Xa[900:]))
Xa, ya = X.astype("float32"), y.astype("float32")
go("split, gamma score, float32", lambda: SplitConformalRegressor(LinearRegression(), confidence_level=0.9, conformity_score="gamma", prefit=False).fit(Xa[:500], ya[:500]).conformalize(Xa[500:900], ya[500:900]).predict_interval(Xa[900:]))
go("cross-conformal, float32", lambda: CrossConformalRegressor(LinearRegression(), confidence_level=0.9).fit_conformalize(Xa[:900], ya[:900]).predict_interval(Xa[900:]))
go("split, float32, AbsoluteConformityScore(consistency_check=False)", lambda: SplitConformalRegressor(LinearRegression(), confidence_level=0.9, conformity_score=AbsoluteConformityScore(consistency_check=False), prefit=False).fit(Xa[:500], ya[:500]).conformalize(Xa[500:900], ya[500:900]).predict_interval(Xa[900:]))
import pandas as pd
df = pd.DataFrame(X, columns=list("abcd")).astype("float32"); ys = pd.Series(y).astype("float32")
go("split, pandas float32 frame (as read from many parquet files)", lambda: SplitConformalRegressor(LinearRegression(), confidence_level=0.9, prefit=False).fit(df[:500], ys[:500]).conformalize(df[500:900], ys[500:900]).predict_interval(df[900:]))
