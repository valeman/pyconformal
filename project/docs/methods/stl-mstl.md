<!-- requires:  -->
# STL and MSTL decomposition

Fast trend and seasonal decomposition, for one or several seasonal periods.

| | |
|---|---|
| **Use it when** | You want to remove seasonality before forecasting, or look at the parts. |
| **Guarantee** | Not a conformal method. |
| **In the library** | `stl`, `mstl` |
| **Source** | Cleveland et al. (1990); Bandara, Hyndman and Bergmeir (2021) |
| **Example needs** | pyconformal |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import mstl

rng = np.random.default_rng(0)
t = np.arange(24 * 7 * 8)
y = 0.01 * t + 3 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 168) + rng.normal(scale=0.3, size=t.size)

parts = mstl(y, periods=[24, 168])                 # hourly data: day and week
print(parts.trend.shape, parts.seasonal[24].shape, parts.remainder.shape)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
