<!-- requires: sklearn -->
# Adaptive conformal inference (ACI)

Intervals that widen and narrow online so the long-run miss rate stays on target, whatever the data does.

| | |
|---|---|
| **Use it when** | The data drifts and there is no exchangeability to rely on. |
| **Guarantee** | Long-run miss rate within a known bound of alpha, for any sequence. The level is not clipped, so intervals can be infinite. |
| **In the library** | `AdaptiveConformalInference` |
| **Source** | Gibbs and Candès (2021) |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np

rng = np.random.default_rng(0)
T = 1000
actuals = np.sin(np.arange(T) / 20) + rng.normal(scale=0.3 + np.arange(T) / 1000, size=T)
forecasts = np.sin(np.arange(T) / 20)          # any forecast, made before each actual

from pyconformal import AdaptiveConformalInference

aci = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
result = aci.run(y_pred=forecasts[:-1], y=actuals[:-1])     # replay the history
print("coverage:", result.coverage)
interval = aci.predict_interval(y_pred=forecasts[-1:])      # then go live
aci.update(y_pred=forecasts[-1:], y=actuals[-1:])
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
