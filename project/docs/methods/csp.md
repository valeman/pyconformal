<!-- requires:  -->
# Conformal Seasonal Pools

A training-free probabilistic forecaster for seasonal series.

| | |
|---|---|
| **Use it when** | You need forecast intervals from the history alone, with no model to train. |
| **Guarantee** | Calibrated on held-out past values; see the user guide for the conditions. |
| **In the library** | `ConformalSeasonalPool` |
| **Example needs** | pyconformal |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import ConformalSeasonalPool

rng = np.random.default_rng(0)
t = np.arange(24 * 60)
history = 10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(scale=0.5, size=t.size)

csp = ConformalSeasonalPool(random_state=0).fit(history, seasonal_period=24)
forecast = csp.predict(H=48, alpha=0.05, n_samples=100)
print(forecast.lower.shape, forecast.upper.shape)  # (48,) each: central 95% interval
median = forecast.quantiles[0.5]
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
