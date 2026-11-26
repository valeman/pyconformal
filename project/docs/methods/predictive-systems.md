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

## With weights (covariate shift)

From version 0.1.6. Give each calibration point a weight in `calibrate`, and each test point its weight in every `predict_*` call. If the weights are the likelihood ratio between the test and the calibration distribution of the inputs, the distributions stay calibrated on the shifted data (Jonkers, Van Wallendael, Duchateau and Van Hoecke, 2024). With estimated weights that holds approximately; with other weights, such as similarity or recency, there is no guarantee.

```python
import numpy as np
from pyconformal import ConformalPredictiveSystem

rng = np.random.default_rng(0)
n = 20_000
x_cal, x_test = rng.normal(0, 1, n), rng.normal(1, 1, n)             # the test inputs have moved
noise = lambda x: (1 + np.abs(x)) * rng.normal(size=x.size)          # and the outcome is noisier there
y_cal, y_test = noise(x_cal), noise(x_test)
ratio = lambda x: np.exp(x - 0.5)                                    # density of N(1, 1) over N(0, 1)

cps = ConformalPredictiveSystem().calibrate(y_pred=np.zeros(n), y=y_cal, weights=ratio(x_cal))
lo, hi = cps.predict_interval(y_pred=np.zeros(n), alpha=0.1, weights=ratio(x_test)).T
print(np.mean((lo <= y_test) & (y_test <= hi)))                      # 0.894; 0.838 without weights
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
