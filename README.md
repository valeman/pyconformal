# pyconformal

Conformal prediction for Python: prediction intervals and prediction sets with
finite-sample coverage guarantees, for any model.

> **Status: early development (0.0.1).** The API will change before 1.0.

## Install

```bash
pip install pyconformal
```

## Regression: prediction intervals

```python
from sklearn.ensemble import RandomForestRegressor
from pyconformal import SplitConformalRegressor

model = SplitConformalRegressor(RandomForestRegressor(), alpha=0.1)
model.fit(X_train, y_train)        # train the model
model.calibrate(X_cal, y_cal)      # held-out calibration data
intervals = model.predict_interval(X_test)   # shape (n, 2): lower, upper
```

## Classification: prediction sets

```python
from sklearn.linear_model import LogisticRegression
from pyconformal import SplitConformalClassifier

model = SplitConformalClassifier(LogisticRegression(), alpha=0.1)
model.fit(X_train, y_train)
model.calibrate(X_cal, y_cal)
sets = model.predict_set(X_test)   # boolean (n, n_classes), columns = model.classes_
```

Already have a trained model? Pass `prefit=True` and skip `fit`.

## Probability calibration: Venn-Abers

```python
from pyconformal import VennAbersCalibrator

model = VennAbersCalibrator(classifier)      # binary or multiclass
model.fit(X_train, y_train)
model.calibrate(X_cal, y_cal)
proba = model.predict_proba(X_test)          # calibrated probabilities
p0_p1 = model.predict_p0_p1(X_test)          # binary: the Venn-Abers pair per sample
```

Works on raw scores too: `VennAbers().fit(scores_cal, y_cal).predict_proba(scores_test)`.
Install `pyconformal[fast]` to use the compiled kernel (numba); results are
identical with or without it.


## The guarantee

If the calibration data and the test point are exchangeable (for example,
i.i.d.), the interval or set contains the true value with probability at least
`1 - alpha`. The guarantee is marginal (on average over test points), not
conditional on each input. It holds for any model and any data distribution.

The calibration data must not be used to train the model. If the calibration
set is too small for the chosen `alpha` (fewer than `1/alpha - 1` points), the
intervals are infinite, which is the only way to keep the guarantee.

## Licence

MIT. Copyright (c) 2026 Valery Manokhin.
