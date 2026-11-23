# pyconformal

Conformal prediction for regression, classification, probability calibration and time series, behind one interface. The only required dependency is NumPy.

```bash
pip install "pyconformal[fast]"
```

```python
from pyconformal import SplitConformalRegressor

cp = SplitConformalRegressor(model, alpha=0.1)   # any fitted model with .predict
cp.calibrate(X_cal, y_cal)                       # data the model was not trained on
cp.predict_interval(X_test)                      # (n, 2): lower and upper, 90% coverage
```

## Where to go

| I want to | Page |
|---|---|
| See every method in the library | [What's inside](whats-inside.md) |
| Copy a minimal example for one method | [What's inside](whats-inside.md), then click the method |
| Get coverage inside each category (Mondrian) | [Mondrian regression](methods/mondrian.md) · [Mondrian classification](methods/mondrian-classification.md) |
| Know what has to be installed | [FAQ](faq.md) |
| Read the long walk-through | [User guide](guide.md) |
| See results on real series | [Case studies](case_studies.md) |
| Look up an argument | [API reference](api.md) |

## The pattern is always the same

1. `fit` the model (skip it if the model is already trained).
2. `calibrate` on data the model has not seen.
3. `predict_interval` or `predict_set`.

Every method also works from predictions alone (`y_pred=`, `proba=`, `quantiles=`), so the model can live anywhere: another library, another machine, a database column.

Every example on this site is run by the test suite on each change, so what you copy works.
