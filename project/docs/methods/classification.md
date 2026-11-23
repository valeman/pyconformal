<!-- requires: sklearn -->
# Split conformal classification (LAC, APS, RAPS, SAPS)

Prediction sets that contain the true class with the chosen probability.

| | |
|---|---|
| **Use it when** | You have a classifier with probabilities and a calibration set. |
| **Guarantee** | At least 1 - alpha coverage on average. |
| **In the library** | `SplitConformalClassifier`, `LAC`, `APS`, `RAPS`, `SAPS` |
| **Source** | Sadinle et al. (2019); Romano et al. (2020); Angelopoulos et al. (2021); Huang et al. (2024) |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=3000, n_features=10, n_informative=6, n_classes=4, random_state=0)
X_train, y_train = X[:1000], y[:1000]
X_cal, y_cal = X[1000:2000], y[1000:2000]
X_test, y_test = X[2000:], y[2000:]
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

from pyconformal import SplitConformalClassifier, RAPS

cp = SplitConformalClassifier(model, alpha=0.1, score="aps", random_state=0)   # "lac", "aps", "raps", "saps"
cp.calibrate(X_cal, y_cal)
sets = cp.predict_set(X_test)                      # boolean, shape (n, n_classes)
print("coverage:", sets[np.arange(len(y_test)), y_test].mean(), "mean size:", sets.sum(axis=1).mean())

# A score with its own settings, and from probabilities only:
cp = SplitConformalClassifier(alpha=0.1, score=RAPS(penalty=0.1, k_reg=1), random_state=0)
cp.calibrate(proba=model.predict_proba(X_cal), y=y_cal)
sets = cp.predict_set(proba=model.predict_proba(X_test))
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
