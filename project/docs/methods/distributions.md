<!-- requires: sklearn -->
# When the outcome is a whole distribution

Conformal sets for vector-valued predictions, such as the coefficients of a predicted distribution.

| | |
|---|---|
| **Use it when** | Each case's outcome is itself a distribution (seen through samples), not one number. |
| **Guarantee** | The ball contains the true coefficient vector with probability at least 1 - alpha. |
| **In the library** | `ConformalDistributionRegressor`, `cosine_coefficients`, `cosine_cdf`, `cosine_density` |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from sklearn.linear_model import Ridge
from pyconformal import ConformalDistributionRegressor, cosine_coefficients

rng = np.random.default_rng(0)
shift = rng.uniform(0.2, 0.8, 600)
X_samples = np.clip(shift[:, None] + rng.normal(scale=0.08, size=(600, 200)), 0, 1)   # 600 cases, 200 points each
Y_samples = np.clip(shift[:, None] ** 2 + rng.normal(scale=0.08, size=(600, 200)), 0, 1)

A_in = cosine_coefficients(X_samples, n_terms=10)
A_out = cosine_coefficients(Y_samples, n_terms=10)
train, cal, test = slice(0, 300), slice(300, 500), slice(500, 600)

cd = ConformalDistributionRegressor(Ridge(), alpha=0.1)
cd.fit(A_in[train], A_out[train]).calibrate(A_in[cal], A_out[cal])
centre, radius = cd.predict_ball(A_in[test])
print("coverage:", cd.contains(A_out[test], A_in[test]).mean())
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
