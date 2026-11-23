<!-- requires: sklearn lightgbm -->
# Quantile models from CatBoost, XGBoost and LightGBM

One wrapper that makes any of the three libraries predict a grid of quantiles.

| | |
|---|---|
| **Use it when** | You already use one of these libraries. Only the one you name has to be installed. |
| **Guarantee** | None on its own; add a conformal method on top. |
| **In the library** | `QuantileBooster` |
| **Example needs** | pyconformal, scikit-learn, LightGBM, CatBoost or XGBoost (only the one you name) |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=1500, n_features=5, noise=20.0, random_state=0)
X_train, y_train = X[:500], y[:500]
X_cal, y_cal = X[500:1000], y[500:1000]
X_test, y_test = X[1000:], y[1000:]
model = LinearRegression().fit(X_train, y_train)

from pyconformal import QuantileBooster, ConformalInterquantileRegressor

qb = QuantileBooster("lightgbm", levels=19, n_estimators=100, verbose=-1)   # or "catboost", "xgboost"
cir = ConformalInterquantileRegressor(qb, plus=True, alpha=0.1)
cir.fit(X_train, y_train).calibrate(X_cal, y_cal)
intervals = cir.predict_interval(X_test)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
