<!-- requires: sklearn -->
# Mondrian classification: by category or by class

Prediction sets with one threshold per category (`groups=`) or per class (`class_conditional=True`).

| | |
|---|---|
| **Use it when** | Coverage must hold inside each category, or for each true class (for example a rare disease class). |
| **Guarantee** | At least 1 - alpha coverage within every category, or within every class. |
| **In the library** | `SplitConformalClassifier` |
| **Source** | Vovk, Gammerman and Shafer (2005) |
| **Example needs** | pyconformal, scikit-learn |

!!! note
    `groups=` on the classifier is new after version 0.1.3. Until the next release, install the development version: `pip install "pyconformal @ git+https://github.com/valeman/pyconformal"`. `class_conditional=True` works in 0.1.3.

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

from pyconformal import SplitConformalClassifier

groups = np.where(X[:, 0] > 0, "north", "south")   # any category per sample
g_cal, g_test = groups[1000:2000], groups[2000:]

cp = SplitConformalClassifier(model, alpha=0.1, score="aps")
cp.calibrate(X_cal, y_cal, groups=g_cal)
sets = cp.predict_set(X_test, groups=g_test)       # boolean, shape (n, n_classes)
print(cp.group_thresholds())

# One threshold per class instead (label-conditional):
cc = SplitConformalClassifier(model, alpha=0.1, class_conditional=True).calibrate(X_cal, y_cal)
sets = cc.predict_set(X_test)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
