<!-- requires: sklearn -->
# DtACI, AgACI, Conformal PID and NexCP

Four more online methods with the same interface as ACI.

| | |
|---|---|
| **Use it when** | You do not want to pick ACI's step size by hand (DtACI, AgACI), want faster reaction (Conformal PID), or want recent data to count more (NexCP). |
| **Guarantee** | DtACI and Conformal PID: long-run coverage for any sequence. NexCP: coverage gap bounded by the drift. |
| **In the library** | `DtACI`, `AgACI`, `ConformalPID`, `NonexchangeableConformal` |
| **Source** | Gibbs and Candès (2022); Zaffran et al. (2022); Angelopoulos, Candès and Tibshirani (2023); Barber et al. (2023) |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np

rng = np.random.default_rng(0)
T = 1000
actuals = np.sin(np.arange(T) / 20) + rng.normal(scale=0.3 + np.arange(T) / 1000, size=T)
forecasts = np.sin(np.arange(T) / 20)          # any forecast, made before each actual

from pyconformal import DtACI, AgACI, ConformalPID, NonexchangeableConformal

for method in (DtACI(alpha=0.1), AgACI(alpha=0.1), ConformalPID(alpha=0.1), NonexchangeableConformal(alpha=0.1, rho=0.99)):
    result = method.run(y_pred=forecasts, y=actuals)
    print(type(method).__name__, "coverage:", round(float(result.coverage), 3))
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
