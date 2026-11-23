<!-- requires: sklearn -->
# Conformalized quantile regression (CQR)

Shifts a lower and an upper predicted quantile until they cover.

| | |
|---|---|
| **Use it when** | You have a model that predicts two quantiles. |
| **Guarantee** | At least 1 - alpha coverage on average. |
| **In the library** | `ConformalizedQuantileRegressor` |
| **Source** | Romano, Patterson and Candès (2019) |
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

# Any quantile model works. Here: the fitted mean plus normal quantiles.
from statistics import NormalDist
levels = np.linspace(0.05, 0.95, 19)
z = np.array([NormalDist().inv_cdf(p) for p in levels])
spread = np.std(y_train - model.predict(X_train))
Q_cal = model.predict(X_cal)[:, None] + spread * z
Q_test = model.predict(X_test)[:, None] + spread * z

from pyconformal import ConformalizedQuantileRegressor

q_cal, q_test = Q_cal[:, [0, -1]], Q_test[:, [0, -1]]      # 5% and 95% quantiles
cqr = ConformalizedQuantileRegressor(alpha=0.1)
cqr.calibrate(quantiles=q_cal, y=y_cal)
intervals = cqr.predict_interval(quantiles=q_test)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
