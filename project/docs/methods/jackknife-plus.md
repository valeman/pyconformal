<!-- requires: sklearn -->
# Jackknife+ and CV+

Intervals without setting data aside for calibration.

| | |
|---|---|
| **Use it when** | Data is scarce and you cannot afford a calibration set. |
| **Guarantee** | At least 1 - 2 alpha coverage in the worst case; close to 1 - alpha in practice. |
| **In the library** | `CrossConformalRegressor` |
| **Source** | Barber, Candès, Ramdas and Tibshirani (2021) |
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

from pyconformal import CrossConformalRegressor

cv = CrossConformalRegressor(LinearRegression(), alpha=0.1, cv=10, random_state=0)   # CV+
cv.fit(X[:1000], y[:1000])
intervals = cv.predict_interval(X_test)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
