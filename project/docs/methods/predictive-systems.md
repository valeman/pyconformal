<!-- requires: sklearn -->
# Conformal predictive systems

A whole calibrated predictive distribution for each prediction, not one interval.

| | |
|---|---|
| **Use it when** | You need quantiles, probabilities of exceeding a level, or CRPS. |
| **Guarantee** | The distribution is calibrated in probability for exchangeable data. |
| **In the library** | `ConformalPredictiveSystem` |
| **Source** | Vovk, Shen, Manokhin and Xie (2019) |
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

from pyconformal import ConformalPredictiveSystem

cps = ConformalPredictiveSystem(model, alpha=0.1).calibrate(X_cal, y_cal)
intervals = cps.predict_interval(X_test)
medians = cps.predict_quantile(X_test, 0.5)
p_below = cps.predict_cdf(X_test, 0.0)             # P(y <= 0) for each sample
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
