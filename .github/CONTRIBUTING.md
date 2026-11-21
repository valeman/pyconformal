# Contributing to pyconformal

Contributions are welcome: new methods, bug reports, benchmarks, documentation.

## Where things go

- **A method you want added:** open a [method request](https://github.com/valeman/pyconformal/issues/new?template=method_request.yml). Vote on existing requests with a thumbs-up on the issue; the most-wanted come first.
- **A bug:** open a [bug report](https://github.com/valeman/pyconformal/issues/new?template=bug_report.yml) with a short script that reproduces it.
- **An idea or a question:** start a [discussion](https://github.com/valeman/pyconformal/discussions).
- **Something to work on:** issues labelled `good first issue` and `help wanted` are open to anyone. Comment on the issue before you start so two people do not do the same work.

## What a new method needs

Every method in the package meets the same three rules. A pull request for a new method is merged when it does too.

1. **Exact.** Thresholds come from `conformal_quantile`. Do not compute a quantile rank by hand.
2. **Proved.** Tests check the method three ways:
   - against a short reference written straight from the paper;
   - against existing code, where there is any, on the same inputs;
   - against the guarantee itself, on small calibration sets.
3. **Fast.** Written for the speed the algorithm allows, with a NumPy path that gives the same answer when Numba is not installed. Include a benchmark script in `project/benchmarks/` if you claim a speed-up.

It also follows the one interface: `fit`, `calibrate`, `predict_*`, and `update` or `run` for online methods; `alpha` for the level; the estimator is optional.

## Licences

Do not copy code from a project whose licence does not allow it (GPL, LGPL, or no licence at all). Write from the paper and compare outputs by running the other code.

## Setup

```
git clone https://github.com/valeman/pyconformal
cd pyconformal
pip install -e ".[test,fast]"
pytest
```

All tests must pass with and without Numba.
