<!-- requires:  -->
# The conformal quantile, on its own

The exact split-conformal threshold for a set of scores, in exact arithmetic.

| | |
|---|---|
| **Use it when** | You are building your own method and want the rank computed correctly. |
| **Guarantee** | The k-th smallest score with k = ceil((n + 1)(1 - alpha)); infinite when no such rank exists. |
| **In the library** | `conformal_quantile` |
| **Example needs** | pyconformal |

## Minimal example

Copy, paste, run.

```python
import numpy as np
from pyconformal import conformal_quantile

scores = np.abs(np.random.default_rng(0).normal(size=99))
q = conformal_quantile(scores, alpha=0.1)          # the 90th smallest of 99
print(q)
```

## More

- Full argument list: [API reference](../api.md)
- Longer walk-through and results on real data: [user guide](../guide.md)
- All methods: [what's inside](../whats-inside.md)
