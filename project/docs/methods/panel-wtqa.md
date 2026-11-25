<!-- requires: -->
# Panel data: W-TQA and TQA

Prediction sets for one unit of a panel (a store, a household, a stock, a patient), calibrated on the other units in the same round.

| | |
|---|---|
| **Use it when** | Many units are observed over the same rounds, the peers' outcomes arrive on time, and the target's own outcome arrives late, sometimes, or never. |
| **Guarantee** | On the rounds where the target's outcome is revealed, the long-run miss rate is within a known bound of alpha for any data (with the default `finite=False`). No method can guarantee coverage at every round without assuming the target resembles its peers. |
| **In the library** | `WeightedTQA`, `PanelResult` |
| **Source** | Tu and Giesecke (2026), arXiv:2605.17705; Lin, Trivedi and Sun (2022) for TQA |
| **Example needs** | pyconformal |

!!! note
    Needs pyconformal 0.1.5 or later.

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import WeightedTQA

rng = np.random.default_rng(0)
T, N, M = 300, 200, 40                     # rounds, calibration units, target units
hard = rng.random(N + M) < 0.2             # one unit in five is three times noisier
X = np.where(hard, 2.0, 0.0)[None, :, None] + rng.normal(size=(T, N + M, 2))
y_pred = np.zeros((T, N + M))              # any forecast
y = rng.normal(size=(T, N + M)) * np.where(hard, 3.0, 1.0)

tqa = WeightedTQA(alpha=0.1)               # bandwidth=0.6, gamma=0.01
res = tqa.run(
    y_pred_cal=y_pred[:, :N], y_cal=y[:, :N], X_cal=X[:, :N],
    y_pred_target=y_pred[:, N:], y_target=y[:, N:], X_target=X[:, N:],
    revealed=rng.random(T) < 0.2,          # the target's outcome is seen on 20% of rounds
)
print(res.intervals.shape)                 # (300, 40, 2)
print("coverage:", round(res.coverage, 3))
print("worst 10% of units:", round(res.tail_coverage(0.1), 3))

# Split conformal on the same cross-section, for comparison:
split = WeightedTQA(alpha=0.1, bandwidth=np.inf, gamma=0.0).run(
    y_pred_cal=y_pred[:, :N], y_cal=y[:, :N],
    y_pred_target=y_pred[:, N:], y_target=y[:, N:])
print("split, worst 10% of units:", round(split.tail_coverage(0.1), 3))
```

## Settings

| Setting | Meaning |
|---|---|
| `bandwidth=np.inf` | Equal weights: TQA. |
| `gamma=0` | Fixed level: the similarity weights alone. |
| `finite=True` | The paper's Algorithm 1: an infinite set is replaced by the round's largest calibration score. Bounded sets, but the any-data bound on the miss rate is lost. |
| `start`, `threshold`, `update` | The same method one round at a time, for live use. |

`X` should hold covariates that describe the unit itself (size, price level, history statistics). Calendar variables that are the same for every unit add nothing to the weights.

## Checked against the paper

On the M5 panel the package reproduces the paper's table: W-TQA average coverage 0.925 (paper 0.925), coverage of the worst 10% of units 0.843 (paper 0.847), against 0.566 (paper 0.579) for split conformal. On the SGSC household electricity panel it does the same (0.857 against 0.858). Details: [benchmarks/wtqa](https://github.com/valeman/pyconformal/tree/main/project/benchmarks/wtqa).

## More

- Full argument list: [API reference](../api.md)
- All methods: [what's inside](../whats-inside.md)
