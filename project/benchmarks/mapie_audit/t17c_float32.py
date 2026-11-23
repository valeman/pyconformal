import numpy as np, warnings; warnings.simplefilter("ignore")
from sklearn.linear_model import LinearRegression
from mapie.regression import SplitConformalRegressor
R = 200
print("Share of 200 random datasets on which SplitConformalRegressor(LinearRegression) raises, by input type and by the size of y")
for off in (0.0, 1.0, 50.0, 1000.0):
    for xt, yt_ in (("float64", "float64"), ("float32", "float64"), ("float64", "float32"), ("float32", "float32")):
        bad = 0; msg = ""
        for r in range(R):
            rng = np.random.default_rng(r); X = rng.normal(size=(1200, 4)); y = off + X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(size=1200)
            Xa, ya = X.astype(xt), y.astype(yt_)
            try: SplitConformalRegressor(LinearRegression(), confidence_level=0.9, prefit=False).fit(Xa[:500], ya[:500]).conformalize(Xa[500:900], ya[500:900]).predict_interval(Xa[900:])
            except Exception as e: bad += 1; msg = f"{type(e).__name__}: {' '.join(str(e).split())[:150]}"
        print(f"  y centred at {off:6.0f}, X {xt}, y {yt_}: raises on {bad/R:.2f} of datasets  {msg}")
