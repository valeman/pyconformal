# pyconformal

**Conformal prediction that is exact, fast, and covers both tabular data and time series.**

[![tests](https://github.com/valeman/pyconformal/actions/workflows/tests.yml/badge.svg)](https://github.com/valeman/pyconformal/actions/workflows/tests.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](project/CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.12%20%7C%203.14-blue)](pyproject.toml)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)
[![Dependencies: NumPy only](https://img.shields.io/badge/dependencies-NumPy%20only-informational)](pyproject.toml)
[![GitHub stars](https://img.shields.io/github/stars/valeman/pyconformal?style=social)](https://github.com/valeman/pyconformal/stargazers)

<p>
  <a href="https://buymeacoffee.com/valeman"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="50" width="213"></a>
  &nbsp;&nbsp;
  <a href="https://buymeacoffee.com/valeman"><img src="project/assets/buymeacoffee_qr.png" alt="QR code for buymeacoffee.com/valeman" width="90"></a>
</p>

Prediction intervals, prediction sets, predictive distributions and calibrated
probabilities with finite-sample guarantees, for any model. About thirty
methods behind one interface. NumPy is the only requirement. MIT licence.

```python
from pyconformal import SplitConformalRegressor

cp = SplitConformalRegressor(model, alpha=0.1)   # any scikit-learn-style model
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)
cp.predict_interval(X_test)                      # covers at least 90% of the time
```

No model object? Pass its outputs: `cp.calibrate(y_pred=pred_cal, y=y_cal)`.

**[Install](#install) · [What you get](#what-you-get) · [Quick tour](#one-interface) · [Why it exists](#why-it-exists) · [Speed](#speed-method-by-method) · [User guide](project/docs/guide.md)**

## Install

```bash
pip install "pyconformal[fast] @ git+https://github.com/valeman/pyconformal"
```

This installs version 0.1.0 with numba for speed; drop `[fast]` for a NumPy-only install. The PyPI release of 0.1.0 is coming; until then `pip install pyconformal` gives an early placeholder.

Python 3.10 or later.

## What you get

| | Methods |
|---|---|
| **Regression** | Split conformal (normalized, Mondrian, weighted) · jackknife+ and CV+ · CQR · CTI · CIR and CIR+ · a parallel quantile forest with out-of-bag calibration |
| **Classification** | LAC · APS · RAPS · SAPS, randomised or not, marginal or per class |
| **Probability calibration** | Venn-Abers and cross Venn-Abers, binary and multiclass |
| **Predictive distributions** | Conformal predictive systems: quantiles, distribution function, intervals, exact CRPS |
| **Time series, online** | ACI · DtACI · AgACI · NexCP · conformal PID |
| **Time series, ensembles** | EnbPI · SPCI |
| **Time series, several steps ahead** | Multi-step split conformal · AcMCP · conformal seasonal pools |
| **Decomposition** | STL and MSTL, matching R to 14 decimal places |

## One interface

The same four verbs for every method: `fit` trains the model, `calibrate`
uses held-out data, `predict_*` gives the output, and `update` (or `run`, for
a whole history) serves online methods. `alpha` is the miscoverage level:
0.1 means 90%.

```python
from pyconformal import (SplitConformalClassifier, VennAbers,
                         ConformalPredictiveSystem, AdaptiveConformalInference)

# Prediction sets
cp = SplitConformalClassifier(clf, alpha=0.1, score="aps").calibrate(X_cal, y_cal)
sets = cp.predict_set(X_test)

# Calibrated probabilities with a validity guarantee
proba = VennAbers(clf).calibrate(X_cal, y_cal).predict_proba(X_test)

# A full predictive distribution per sample
cps = ConformalPredictiveSystem(model).calibrate(X_cal, y_cal)
cps.predict_quantile(X_test, [0.05, 0.5, 0.95]); cps.predict_cdf(X_test, 100.0)

# Time series: replay a history, then go live
aci = AdaptiveConformalInference(alpha=0.1)
aci.run(y_pred=forecasts, y=actuals).coverage
aci.predict_interval(y_pred=next_forecast)
aci.update(y_pred=next_forecast, y=next_actual)
```

Results are plain NumPy arrays. The **[user guide](project/docs/guide.md)** has a
section and an example for every method.

## Why it exists

A conformal method makes one promise: a 90% interval covers at least 90% of
the time, for any model, at any sample size. We tested the current releases of
the main packages on plain i.i.d. data, where that promise must hold
(20 November 2026).

**Some widely used software breaks the promise.**

| Package | Setting | Target | Measured |
|---|---|---|---|
| statsforecast 2.1.1 | default (2 calibration windows) | 90% | 63–65% coverage |
| mlforecast 1.1.0 | default (2 calibration windows) | 90% | 63–65% coverage |
| Darts 0.48.1 | `cal_length=3` | 90% | 75% coverage |
| conformalpy 1.0.0 | 3 calibration points | 90% | 75% coverage; threshold exact in 124 of 1,240 cases |
| TorchCP 1.2.1 | `SAPS(randomized=False)`, an accurate classifier | 90% | 2.5–36% coverage |
| crepes 0.9.1 | CRPS with difficulty estimates | | 7.19 reported where 2.19 is correct |
| crepes 0.9.1 | 9 calibration points | 90% | infinite interval where a finite one is valid |
| PUNCC 0.9.3 | EnbPI updated after every observation | | crashes |

None of the coverage failures raised a warning. With two calibration errors no
finite 90% interval is valid; at least nine are needed, and the only honest
answer below that is an infinite interval. (The statsforecast behaviour is
[issue #1349](https://github.com/Nixtla/statsforecast/issues/1349). TorchCP's
default, randomised mode is correct. The crepes and PUNCC items do not affect
the coverage of their intervals.)

**MAPIE is correct. Its costs are elsewhere.** It
passed every one of the 84 coverage configurations we ran, and we found no
setting in which it returns an invalid interval. Its problems are speed,
memory, and sets that are larger than they need to be:

| | MAPIE 1.6.0 | pyconformal | Times faster | Output |
|---|---|---|---|---|
| Venn-Abers, 1,000,000 points | 61.9 s, 244 to 360 numerical warnings per fit | 0.33 s, none | **188x** | identical |
| ACI, backtest of 20,000 steps | 8.1 s | 0.003 s | **about 2,000x** | same coverage |
| ACI, one row at a time | 0.57 ms per step | 0.09 ms | **6x to 8x** | same coverage |
| EnbPI, backtest | 5.5 ms per step | 0.012 ms | **400x to 470x** | same coverage |
| EnbPI, one row at a time | 5.8 ms per step | 2.2 ms | **2.5x** | same coverage |
| CV+ prediction, 2,000 to 5,000 training rows | 2 to 5 s | 0.12 to 0.18 s | **17x to 27x** | same coverage |
| CV+, 20,000 training by 20,000 test rows | out of memory on a 7 GB machine | runs; also at 100,000 rows | | |
| CQR, 1,000,000 test points | 19.7 ms | 3.2 ms | **6x** | identical |
| Prediction sets (LAC, APS, RAPS), 10 to 1,000 classes | 0.010 to 1.8 s | 0.001 to 0.36 s | **5x to 10x** | |
| Split regression, 1,000,000 test points | 8.9 ms | 1.9 ms | **4.7x** | identical |

Sets and scope:

| | MAPIE 1.6.0 | pyconformal |
|---|---|---|
| APS at a 90% target, default settings | covers 97.5%, mean set 2.05 classes (18% larger than needed) | covers 89.9%, mean set 1.73 |
| RAPS at a 90% target, default settings | covers 96.9%, mean set 2.11 classes (22% larger than needed) | covers 89.9%, mean set 1.73 |
| RAPS with 1,000 classes and 5,000 calibration points | error: needs two calibration points in every class | runs |
| Time-series methods | 2 (ACI, EnbPI) | 10 |
| Predictive distributions, CTI, CIR | not available | available |

The backtest rows and the one-row-at-a-time rows measure different things: in
a backtest pyconformal calls the model once for the whole history, where
MAPIE's interface calls it at every step. MAPIE's ACI also clips its working
level to between 0 and 1, which the method's proof does not allow for; we have
not measured a case where that changes its coverage.

**Other correct packages are slow in the same way.**

| Task | Other package | pyconformal | Times faster | Output |
|---|---|---|---|---|
| Venn-Abers calibration, 1,000,000 points | 61.5 s (venn-abers) | 0.33 s | **188x** | identical |
| Predictive intervals, 1,000,000 test points | 79 s (crepes) | 0.01 s | **8,100x** | identical |
| Predictive distribution function, 1,000,000 test points | 145 s (crepes) | 0.26 s | **556x** | identical |
| CRPS, 5,000 test points | 0.31 s (crepes) | 0.0005 s | **656x** | identical without difficulty estimates |
| Multi-step forecast intervals, 3,000 steps × 12 horizons | 42 s (R conformalForecast) | 0.01 s | **over 4,000x** | identical |
| AcMCP, 1,000 steps × 6 horizons | 107 s (R conformalForecast) | 2.5 s | **43x** | median difference 0.00001 |
| Conformal PID, per step | 10 to 14 µs (authors' code) | 0.09 to 0.13 µs | **106x to 112x** | identical |
| CTI, 200,000 test points | 0.43 s (authors' code) | 0.023 s | **19x** | identical |
| CIR, 200,000 test points | 1.85 s (authors' code) | 0.12 s | **16x** | identical in the authors' mode |
| MSTL decomposition, 100,000 points | 19.3 s (statsforecast, its defaults) | 0.098 s | **about 200x** (6x to 7x at equal settings) | identical at equal settings |
| Quantile forest, fit on 1,000,000 rows (16 cores) | 215 s (quantile-forest) | 27 s | **8x** | same accuracy |

**No package covers both worlds.** Tabular libraries carry a couple of
time-series methods; forecasting libraries carry a couple of conformal ones.

## Three rules the package is built on

1. **Exact.** One routine computes every threshold, in exact arithmetic.
   How the rank is written matters. Ask for 90% as a confidence of 0.9 and
   `1 - 0.9` is `0.09999999999999998` in floating point; code that truncates
   `(n + 1) * (1 - 0.9)` then takes one rank too high whenever `n + 1` is a
   multiple of ten, and with 9 calibration points it returns an infinite
   interval where a finite one is valid. When no finite interval is valid,
   you get an infinite one, never a narrow one that looks fine.
2. **Proved.** Every method is tested three ways: against a short reference
   written from its paper, against existing code where there is any
   (venn-abers, crepes, MAPIE, PUNCC, R's `forecast` and `conformalForecast`,
   and the CTI and CIR authors' code), and against the guarantee itself on
   small calibration sets. 234 tests, run with warnings as errors.
3. **Fast.** Each method is written for the speed its algorithm allows, with
   optional compiled kernels (numba). The compiled and plain NumPy paths are
   tested to return the same values, so installing numba changes speed and
   nothing else.

## Speed, method by method

Same inputs on both sides, on a 2-core machine with numba. "Authors' code" is
the code released with the paper that proposed the method.

<details>
<summary><b>Show the full table (21 methods)</b></summary>

| Method | Compared with | Times faster |
|---|---|---|
| Venn-Abers | venn-abers, MAPIE | up to 188x (11x to 29x at 10,000 to 100,000 calibration points), same outputs |
| Cross Venn-Abers | venn-abers | 3x |
| Predictive distributions; CRPS | crepes | 64x to 8,000x; 280x to 760x |
| Split regression | PUNCC, crepes, MAPIE | 2x to 5x |
| Normalized / Mondrian split | crepes | 3x / 4x |
| CV+ | MAPIE | 17x to 27x at prediction |
| CQR | MAPIE; PUNCC | 6x; 3x, same intervals |
| CTI | authors' code | 19x, same sets |
| CIR; CIR+ | authors' code | 13x to 16x, same intervals in the authors' mode; 22x |
| APS, RAPS, SAPS | TorchCP, PUNCC, MAPIE | 2x at 10 classes, 3x to 7x at 100 and 1,000 |
| Quantile forest | quantile-forest | fit 6x to 9x; predict 1.6x to 4x (16-core server) |
| MSTL | statsforecast; R forecast | 6x to 7x at equal settings; 5x |
| ACI | authors' code; MAPIE | 286x; about 2,000x in a backtest, 6x to 8x one row at a time |
| DtACI; AgACI | authors' R code | 66x; 4.5x on the update step |
| Conformal PID | authors' code | 106x to 112x, same thresholds |
| NexCP | the definition, recomputed each step | 236x, same thresholds |
| EnbPI | MAPIE; PUNCC | 400x to 470x in a backtest, about 2.5x one row at a time; 23x |
| SPCI | the paper's loop | 9x |
| Multi-step split | R conformalForecast | over 4,000x, same intervals |
| AcMCP | R conformalForecast | 43x with the scorecaster, about 5,000x without |
| CSP | csp-forecaster | 2x to 21x |

</details>

Read these with three things in mind. **Backtest and live use are different
measurements:** in a backtest pyconformal calls the model once for the whole
history, where MAPIE's interface calls it at every step; fed one row at a
time, the model call dominates in both. **Some gains depend on the data:**
Venn-Abers is 3 to 6 times faster, not 188, on scores rounded to two
decimals. **Small things are small:** split conformal takes milliseconds in
every package.

## Sets that are no larger than needed

Being valid is half the job; the other half is not wasting width. On the same
probabilities (10 classes, 90% target):

| Method | Package | Coverage | Mean set size |
|---|---|---|---|
| APS | pyconformal | 0.900 | 1.73 |
| APS (default) | MAPIE 1.6.0 | 0.975 | 2.05 |

On a skewed regression target, mean interval length at 90% coverage was 5.59
for CQR, 5.18 for CIR, 5.02 for CIR+ and 4.95 for CTI. All four are in the
package, on one quantile forest.

## The guarantee, and its limits

If the calibration data and the test point are exchangeable (for example,
i.i.d.), the interval or set contains the true value with probability at least
`1 - alpha`. This is on average over test points, not for each input
separately, and the calibration data must not have been used to train the
model.

Time series are not exchangeable. The online methods (ACI and its relatives,
conformal PID, AcMCP) guarantee the long-run miss rate on any sequence; EnbPI,
SPCI and conformal seasonal pools are approximate, and the guide says so where
it matters. All of our coverage tests use synthetic data, and the timings come
from one machine.

## More

- [User guide](project/docs/guide.md): every method, with examples.
- [Changelog](project/CHANGELOG.md) · [How to cite](project/CITATION.cff)

## Learn conformal prediction

- **[Awesome Conformal Prediction](https://github.com/valeman/awesome-conformal-prediction)**:
  the curated list of papers, tutorials, videos and libraries, maintained by
  the author of this package.
- **Books by the author**
  - [Applied Conformal Prediction](https://valeman.gumroad.com/l/appliedconformalprediction)
  - [Practical Guide to Applied Conformal Prediction in Python](https://github.com/PacktPublishing/Practical-Guide-to-Applied-Conformal-Prediction) (Packt, 2023)
  - [Mastering Modern Time Series Forecasting](https://valeman.gumroad.com/l/MasteringModernTimeSeriesForecasting)
- **Papers behind two of the methods**:
  [Classifier Calibration at Scale](https://arxiv.org/abs/2601.19944) (why
  Venn-Abers) and
  [Conformal Seasonal Pools](https://arxiv.org/abs/2605.03789).

## Support the project

If pyconformal saves you time, a star on GitHub helps other people find it.
You can also buy me a coffee:

<p>
  <a href="https://buymeacoffee.com/valeman"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="60" width="255"></a>
  &nbsp;&nbsp;
  <a href="https://buymeacoffee.com/valeman"><img src="project/assets/buymeacoffee_qr.png" alt="QR code for buymeacoffee.com/valeman" width="120"></a>
</p>

## Licence and disclaimer

MIT. Copyright (c) 2026 Valery Manokhin. Use it anywhere, including in
closed-source products.

The software is provided "as is", without warranty of any kind, express or
implied, including the warranties of merchantability, fitness for a particular
purpose and non-infringement. The author accepts no liability for any claim,
damages or other loss arising from its use. The full terms are in
[LICENSE](LICENSE).

The coverage guarantees described here are mathematical statements that hold
under stated assumptions, such as exchangeable data and a calibration set kept
apart from training. They are not a guarantee about your data, your model or
any particular decision. You are responsible for checking that the assumptions
fit your use and for validating results before relying on them, especially in
medical, financial, legal or safety-critical settings. Nothing here is
professional advice.

Comparisons with other packages describe the versions and settings named, as
measured on the dates given. Other packages change; check current versions
before relying on a comparison.
