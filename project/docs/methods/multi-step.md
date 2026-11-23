<!-- requires: sklearn -->
# Multi-step forecasts: MSCP and AcMCP

Intervals for forecasts several steps ahead, one calibration per horizon.

| | |
|---|---|
| **Use it when** | You forecast H steps ahead and need an interval at each step. |
| **Guarantee** | MSCP: per-horizon coverage under exchangeable errors. AcMCP: long-run coverage per horizon. |
| **In the library** | `MultiStepSplitConformal`, `AcMCP` |
| **Source** | Wang and Hyndman |
| **Example needs** | pyconformal, scikit-learn |

## Minimal example

Copy, paste, run.

```python
import numpy as np

rng = np.random.default_rng(0)
T, H = 600, 4
y = np.cumsum(rng.normal(size=T))
# F[t, h - 1]: the forecast for time t made h steps earlier (here: the last value seen)
F = np.column_stack([np.r_[np.full(h, y[0]), y[:-h]] for h in range(1, H + 1)])
next_forecasts = np.full(H, y[-1])

from pyconformal import MultiStepSplitConformal, AcMCP

ms = MultiStepSplitConformal(alpha=0.1)
result = ms.run(y_pred=F, y=y)
print(result.intervals.shape)                      # (T, H, 2)
ahead = ms.predict_interval(y_pred=next_forecasts) # the H steps after the history

ac = AcMCP(alpha=0.1, min_calibration=50)
result = ac.run(y_pred=F, y=y)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
