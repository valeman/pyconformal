<!-- requires: sklearn -->
# Venn-Abers calibration

Calibrated probabilities from any classifier's scores, binary or multi-class.

| | |
|---|---|
| **Use it when** | A model's probabilities are over- or under-confident. |
| **Guarantee** | The pair (p0, p1) contains a perfectly calibrated probability (inductive version). |
| **In the library** | `VennAbers`, `CrossVennAbers` |
| **Source** | Vovk and Petej (2014); Manokhin (2017) |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=3000, n_features=10, random_state=0)
X_train, y_train = X[:1000], y[:1000]
X_cal, y_cal = X[1000:2000], y[1000:2000]
X_test, y_test = X[2000:], y[2000:]

from pyconformal import VennAbers, CrossVennAbers

va = VennAbers(GaussianNB()).fit(X_train, y_train).calibrate(X_cal, y_cal)
proba = va.predict_proba(X_test)                   # calibrated, shape (n, 2)

cva = CrossVennAbers(GaussianNB(), cv=5, random_state=0).fit(X[:2000], y[:2000])   # no separate calibration set
proba = cva.predict_proba(X_test)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
