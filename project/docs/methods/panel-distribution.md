<!-- requires: -->
# Predictive distributions for panels

A full predictive distribution for each unit of a panel, every round: built from the unit's peers in the same round, then corrected with the unit's own outcomes whenever they are revealed.

| | |
|---|---|
| **Use it when** | Many units share the same rounds, you need quantiles or probabilities for each unit (not one interval), and a unit's own outcome arrives late, sometimes, or never. |
| **Guarantee** | On the rounds where a unit's outcome is revealed, every quantile level is right in the long run (bounded errors, any data); quantiles never cross. |
| **In the library** | `PanelPredictiveSystem` |
| **Source** | Base distribution: similarity weights of Tu and Giesecke (2026), arXiv:2605.17705. Online correction: MultiQT, Ding, Gibbs and Tibshirani (2025), arXiv:2512.23671. The combination is ours. |
| **Example needs** | pyconformal |

!!! note
    New after version 0.1.4. Until the next release: `pip install "pyconformal @ git+https://github.com/valeman/pyconformal"`.

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import PanelPredictiveSystem

rng = np.random.default_rng(0)
T, N, M = 300, 200, 40                       # rounds, calibration units, target units
hard = rng.random(N + M) < 0.2               # one unit in five is three times noisier
bias = np.where(rng.random(N + M) < 0.3, 1.0, 0.0)   # and some are forecast with a bias
X = np.where(hard, 2.0, 0.0)[None, :, None] + rng.normal(size=(T, N + M, 2))
y = bias + rng.normal(size=(T, N + M)) * np.where(hard, 3.0, 1.0)
y_pred = np.zeros((T, N + M))                # any forecast

pps = PanelPredictiveSystem()                # 19 levels: 0.05 ... 0.95
res = pps.run(
    y_pred_cal=y_pred[:, :N], y_cal=y[:, :N], X_cal=X[:, :N],
    y_pred_target=y_pred[:, N:], y_target=y[:, N:], X_target=X[:, N:],
    revealed=rng.random(T) < 0.2,            # a target's outcome is seen on 20% of rounds
)
print(res.quantiles.shape)                   # (300, 40, 19): rounds, targets, levels
print("calibration error per unit:", round(res.calibration_error.mean(), 3))
intervals = res.interval(0.1)                # central 90% interval per round and target
p_below_zero = res.cdf(0.0)                  # P(y <= 0) per round and target
```

## On real panels

Against a plain conformal predictive system on the cross-section (30 splits each):

| Panel | Calibration error per unit | Pinball loss | 90% coverage of the worst-served 10% of units |
|---|---|---|---|
| M5 retail sales | 0.123 to 0.030 | 16% lower | 0.57 to 0.88 |
| SGSC household electricity | 0.109 to 0.026 | 4% lower | 0.61 to 0.86 |

Details and scripts: [benchmarks/panel_distribution](https://github.com/valeman/pyconformal/tree/main/project/benchmarks/panel_distribution).

## More

- Intervals only: [W-TQA and TQA](panel-wtqa.md)
- One series: [MultiQT](multiqt.md)
- Full argument list: [API reference](../api.md)
