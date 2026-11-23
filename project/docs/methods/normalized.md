<!-- requires: sklearn -->
# Normalized (locally adaptive) intervals

Intervals that are wider where the model is less sure, using a difficulty estimate `sigma`.

| | |
|---|---|
| **Use it when** | Errors are clearly larger in some regions than in others. |
| **Guarantee** | Same marginal guarantee as split conformal. |
| **In the library** | `SplitConformalRegressor` |
| **Source** | Papadopoulos, Gammerman and Vovk (2008) |
| **Example needs** | pyconformal, scikit-learn |

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

from pyconformal import SplitConformalRegressor

# Any positive difficulty estimate. Here: a second model for the size of the error.
err_model = LinearRegression().fit(np.abs(X_train), np.abs(y_train - model.predict(X_train)))
sigma = lambda Z: np.maximum(err_model.predict(np.abs(Z)), 1e-3)

cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate(X_cal, y_cal, sigma=sigma(X_cal))
intervals = cp.predict_interval(X_test, sigma=sigma(X_test))
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
