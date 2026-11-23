<!-- requires: sklearn -->
# Mondrian (group-conditional) conformal prediction

One threshold per category, so coverage holds inside every category and not only on average.

| | |
|---|---|
| **Use it when** | Coverage must hold for each region, product, device, age band or any other category you define. |
| **Guarantee** | At least 1 - alpha coverage within every category. A category that is new at prediction gets an infinite interval (regression) or every class (classification). |
| **In the library** | `SplitConformalRegressor`, `SplitConformalClassifier`, `ConformalPredictiveSystem` |
| **Source** | Vovk, Gammerman and Shafer (2005) |
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

# Any category per sample: numbers or strings.
groups = (X[:, 0] > 0).astype(int)
g_cal, g_test = groups[500:1000], groups[1000:]

cp = SplitConformalRegressor(model, alpha=0.1)
cp.calibrate(X_cal, y_cal, groups=g_cal)
intervals = cp.predict_interval(X_test, groups=g_test)
for g in (0, 1):
    m = g_test == g
    inside = (y_test[m] >= intervals[m, 0]) & (y_test[m] <= intervals[m, 1])
    print("category", g, "coverage:", inside.mean())
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
