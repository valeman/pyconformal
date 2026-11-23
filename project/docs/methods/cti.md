<!-- requires: sklearn -->
# Conformal thresholded intervals (CTI)

Keeps the shortest interquantile intervals; the result can be a union of intervals.

| | |
|---|---|
| **Use it when** | The outcome may have several modes, or you want the smallest set. |
| **Guarantee** | At least 1 - alpha coverage on average. |
| **In the library** | `ConformalThresholdedIntervals` |
| **Source** | Luo and Zhou (2025) |
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

from pyconformal import ConformalThresholdedIntervals

cti = ConformalThresholdedIntervals(alpha=0.1).calibrate(quantiles=Q_cal, y=y_cal)
kept = cti.predict_set(quantiles=Q_test)           # which interquantile intervals are kept
size = cti.predict_size(quantiles=Q_test)          # total length per sample
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
