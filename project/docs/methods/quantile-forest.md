<!-- requires: sklearn numba -->
# Quantile forest

A fast quantile regression forest, usable on its own or under CQR, CIR and CTI.

| | |
|---|---|
| **Use it when** | You want quantiles from a forest without another library. |
| **Guarantee** | None on its own; add a conformal method on top. |
| **In the library** | `QuantileForest` |
| **Source** | Meinshausen (2006) |
| **Example needs** | pyconformal, scikit-learn, numba (`pip install pyconformal[fast]`) |

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

from pyconformal import QuantileForest, ConformalInterquantileRegressor

forest = QuantileForest(n_estimators=50, random_state=0)
cir = ConformalInterquantileRegressor(forest, alpha=0.1, plus=True)
cir.fit(X_train, y_train)
cir.calibrate(X_cal, y_cal)
intervals = cir.predict_interval(X_test)
quantiles = forest.predict(X_test, quantiles=[0.05, 0.5, 0.95])
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
