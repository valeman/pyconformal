<!-- requires: sklearn -->
# Weighted conformal prediction (covariate shift)

Intervals that stay valid when the test inputs come from a different distribution than the calibration inputs.

| | |
|---|---|
| **Use it when** | You know, or can estimate, how much more likely each input is under the test distribution. |
| **Guarantee** | 1 - alpha coverage on the shifted distribution, if the weights are right. |
| **In the library** | `SplitConformalRegressor` |
| **Source** | Tibshirani, Barber, Candès and Ramdas (2019) |
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

w = lambda Z: np.exp(0.5 * Z[:, 0])                # likelihood ratio test / calibration

cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate(X_cal, y_cal, weights=w(X_cal))
intervals = cp.predict_interval(X_test, weights=w(X_test))
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
