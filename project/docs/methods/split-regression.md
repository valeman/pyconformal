<!-- requires: sklearn -->
# Split conformal regression

Prediction intervals around any regression model.

| | |
|---|---|
| **Use it when** | You have a trained model and a held-out calibration set. |
| **Guarantee** | At least 1 - alpha coverage on average, for exchangeable data, at any sample size. |
| **In the library** | `SplitConformalRegressor` |
| **Source** | Vovk, Gammerman and Shafer (2005); Papadopoulos et al. (2002) |
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

cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate(X_cal, y_cal)
intervals = cp.predict_interval(X_test)            # shape (n, 2): lower, upper
print("coverage:", np.mean((y_test >= intervals[:, 0]) & (y_test <= intervals[:, 1])))

# The same from predictions only, with no model object:
cp = SplitConformalRegressor(alpha=0.1).calibrate(y_pred=model.predict(X_cal), y=y_cal)
intervals = cp.predict_interval(y_pred=model.predict(X_test))
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
