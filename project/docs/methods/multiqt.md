<!-- requires: -->
# MultiQT: calibrated quantiles at many levels, online

Takes quantile (or point) forecasts from any model and adjusts them after every outcome so that each level is right in the long run, even under drift. The quantiles never cross.

| | |
|---|---|
| **Use it when** | You issue quantile forecasts or a predictive distribution for a time series and the data drift. |
| **Guarantee** | With bounded forecast errors, for any sequence: the share of rounds with the outcome at or below the level-q quantile tends to q, at every level; quantiles are ordered every round. |
| **In the library** | `MultiQT`, `QuantileResult` |
| **Source** | Ding, Gibbs and Tibshirani (2025), arXiv:2512.23671 |
| **Example needs** | pyconformal |

!!! note
    New after version 0.1.4. Until the next release: `pip install "pyconformal @ git+https://github.com/valeman/pyconformal"`.

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import MultiQT

rng = np.random.default_rng(0)
T = 2000
y = 0.004 * np.arange(T) + rng.normal(size=T) * (1 + 3 * (np.arange(T) > 1000))   # trend, then noisier

levels = [0.05, 0.25, 0.5, 0.75, 0.95]
base = np.tile(np.quantile(rng.normal(size=500), levels), (T, 1))   # a forecaster that never adapts

res = MultiQT(levels).run(base=base, y=y)
print("base:    ", np.round((y[:, None] <= base).mean(0), 2))        # far from the levels
print("adjusted:", np.round(res.frequencies[0], 2))                  # close to 0.05 ... 0.95
print(res.quantiles.shape)                                           # (2000, 1, 5), ordered
intervals = res.interval(0.1)                                        # central 90%
```

## Settings

| Setting | Meaning |
|---|---|
| `lr="adaptive"` | The paper's step-size rule. Pass a number for a fixed step, in the units of the outcome. |
| `lr_floor=0.1` | Lower limit of the adaptive step, in outcome units. Lower it when errors are much smaller than 1. |
| `run(..., delay=d)` | The outcome of a round is seen d rounds later. |
| `run(..., observed=mask)` | Some outcomes are never seen. |
| Several series | Pass `y` of shape (rounds, series); each series is tracked on its own, in one call. |

Checked against the authors' code: same forecasts to 1e-15, 15 times faster. Details: [benchmarks/multiqt](https://github.com/valeman/pyconformal/tree/main/project/benchmarks/multiqt).

## More

- For panels: [predictive distributions for panels](panel-distribution.md)
- Full argument list: [API reference](../api.md)
