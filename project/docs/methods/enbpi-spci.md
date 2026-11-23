<!-- requires: sklearn -->
# EnbPI and SPCI

Bootstrap ensembles that give intervals without a calibration set, with residuals sliding forward.

| | |
|---|---|
| **Use it when** | A time series with features, and not enough data to hold out a calibration set. |
| **Guarantee** | Approximate coverage under conditions on the errors; no finite-sample guarantee under drift. |
| **In the library** | `EnbPI`, `SPCI` |
| **Source** | Xu and Xie (2021, 2023) |
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

from pyconformal import EnbPI

enb = EnbPI(LinearRegression(), alpha=0.1, n_bootstraps=20, random_state=0).fit(X[:1000], y[:1000])
result = enb.run(X_test, y_test)                   # intervals, residuals updated as actuals arrive
print(result.intervals.shape)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
