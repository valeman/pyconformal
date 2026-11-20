# pyconformal

Conformal prediction for Python: prediction intervals and prediction sets with
finite-sample coverage guarantees, for any model.

> **Status: early development (0.0.1).** The API will change before 1.0.

## Why another conformal prediction package

Conformal prediction makes one promise: a 90% interval covers at least 90% of
the time. We tested the current releases of the main Python packages
(20 November 2026). Every number below comes from a script in
[`benchmarks/`](benchmarks/); run them yourself.

### 1. Popular forecasting libraries break the promise at their defaults

On plain i.i.d. data, where the guarantee must hold:

| Package | Setting | Nominal | Measured coverage |
|---|---|---|---|
| statsforecast 2.1.1 | default (2 calibration windows) | 90% | 63–65% |
| mlforecast 1.1.0 | default (2 calibration windows) | 90% | 63–65% |
| Darts 0.48.1 | `cal_length=3` | 90% | 75% |

None of these raised an error or a warning. With 2 calibration errors no valid
finite 90% interval exists; at least 9 are needed. The statsforecast case was
reported as [issue #1349](https://github.com/Nixtla/statsforecast/issues/1349)
and closed without changing the default.

### 2. MAPIE is correct, but slow and narrow where it matters

We audited MAPIE 1.6.0 on 84 configurations: split, CQR, cross and
jackknife+-after-bootstrap regression, and LAC, APS, RAPS and top-k
classification. There were no coverage failures; six configurations refused to
run with a clear error, which is safe. Its measured weaknesses:

- **Venn-Abers:** the same answers as pyconformal, 3 to 188 times slower
  (60.7 s against 0.48 s for 1,000,000 calibration points), with hundreds of
  numerical warnings on every fit. pyconformal does the whole computation in
  integers with a single division at the end, so it emits none.
- **Online time series (ACI):** 0.41 ms per step. pyconformal's ACI runs the
  same task in 0.17 to 0.25 microseconds per step, about 1,600 to 2,400 times
  faster, with the same coverage. MAPIE calls the model one row at a time;
  pyconformal never puts the model inside the update loop.
- **APS and RAPS prediction sets are larger than needed:** 95.5% and 96.3%
  coverage at a 90% target.
- **Time series:** two methods only.

### 3. Predictive distributions, hundreds of times faster than crepes

pyconformal's conformal predictive systems return the same quantiles and
distribution-function values as crepes 0.9.1, for standard, normalized and
Mondrian systems:

| Task | Calibration / test points | crepes | pyconformal | Speed-up |
|---|---|---|---|---|
| 90% intervals | 10,000 / 100,000 | 0.62 s | 0.001 s | 600x |
| 90% intervals | 100,000 / 1,000,000 | 79 s | 0.01 s | 8,100x |
| Distribution function at the outcomes | 10,000 / 100,000 | 1.27 s | 0.02 s | 64x |
| CRPS | 5,000 / 5,000 | 0.31 s | 0.0005 s | 656x |

pyconformal never builds the full distribution for each test point: a
quantile is one lookup, the distribution function is a binary search, and
CRPS has a closed form.

Even plain split conformal regression is faster: 1.9 ms for 200,000
calibration and 1,000,000 test points, against 4.2 ms (PUNCC), 6.0 ms (crepes)
and 8.9 ms (MAPIE).

### 4. Prediction sets: right, small and fast

On the same probabilities (10 classes, 90% target; `benchmarks/speed_classification.py`):

| Method | Library | Coverage | Mean set size |
|---|---|---|---|
| APS | pyconformal | 0.900 | 1.73 |
| APS (default) | MAPIE 1.6.0 | 0.975 | 2.05 |
| SAPS, not randomised | pyconformal | 0.906 | 1.90 |
| SAPS, not randomised | TorchCP 1.2.1 | **0.338** | 0.63 |

MAPIE's default APS and RAPS over-cover, so their sets are larger than
needed. TorchCP's SAPS with `randomized=False` scores the top class
differently in calibration and in prediction and loses the guarantee (its
default randomised mode is correct). In pyconformal one routine computes the
scores for calibration and prediction, so the two cannot disagree. It was also
the fastest of the four libraries in every configuration tested, by up to 2
times over the next best (the margin is small with 1,000 classes).

### 5. Mainstream calibration tools are not the best ones

scikit-learn's `CalibratedClassifierCV` offers Platt scaling (its default),
isotonic regression and temperature scaling. It does not offer Venn-Abers, the
calibration method with a validity guarantee. A study of 21 classifiers on the
TabArena binary tasks
([Manokhin and Grønhaug, 2026](https://arxiv.org/abs/2601.19944)) found that
Venn-Abers gave the largest average log-loss reduction of five calibrators,
and that Platt scaling and isotonic regression can make strong modern tabular
models worse. No calibrator won on every dataset. pyconformal ships Venn-Abers
for binary and multiclass problems, up to 188 times faster than existing
implementations.

### 6. What pyconformal does differently

- **Exact by construction.** One audited routine computes every conformal
  threshold in exact arithmetic.
- **Honest when data is short.** If no valid finite interval exists, you get an
  infinite one, never a narrow one that looks fine.
- **Proof for every method.** Each method ships with a test against a
  reference implementation or a brute-force calculation, a statistical test,
  and a speed benchmark.
- **Fast.** Linear-time algorithms, compiled with numba when it is installed,
  identical results without it.
- **Any model.** Methods work on predictions and scores, so scikit-learn,
  gradient boosting, statistical forecasters and foundation models all fit.
- **One library for tabular data and time series.**
- **MIT licence.** Use it anywhere, including in closed-source products.
  (TorchCP is LGPL-3.0; R's conformalForecast is GPL-3.)

Available now: split conformal regression and classification (LAC, APS, RAPS
and SAPS scores, class-conditional sets), conformalized quantile regression
(CQR), conformal thresholded intervals (CTI), conformal interquantile
regression (CIR and CIR+), conformal predictive distributions, Venn-Abers
calibration (binary and multiclass), adaptive conformal inference (ACI) and
conformal seasonal pools (CSP) for time series, and a fast parallel quantile
forest to feed CQR, CTI and CIR (with out-of-bag calibration), jackknife+ and
CV+, normalized, Mondrian and weighted split regression, cross Venn-Abers,
DtACI, AgACI, NexCP, conformal PID, EnbPI and SPCI for time series, and a fast
STL / MSTL decomposition for series with several seasonal periods.

## Install

```bash
pip install pyconformal
```

## Speed against other libraries and the papers' own code

Same inputs on both sides, measured on a 2-core machine with numba installed.
Scripts are in `benchmarks/`.

| Method | Compared with | pyconformal is |
|---|---|---|
| Venn-Abers | venn-abers, MAPIE | 3x to 188x faster, same outputs |
| Cross Venn-Abers | venn-abers | 3x faster (model training included on both sides) |
| ACI | authors' code; MAPIE | 286x; about 2,000x |
| DtACI | authors' R code | 357x from scores to thresholds (66x on the update step alone) |
| AgACI | R code in the DtACI repository, 30 experts | 46x from scores to thresholds (4.5x on the update step alone) |
| Conformal PID | authors' code | 106x to 112x, same thresholds |
| NexCP | the definition recomputed each step | 236x, same thresholds |
| EnbPI | MAPIE; PUNCC | 470x; 23x (PUNCC fails at batch size 1) |
| SPCI | the paper's loop with quantile-forest | 9x |
| CV+ (10 folds) | MAPIE | 17x to 27x at prediction; MAPIE runs out of memory at 20,000 x 20,000 rows |
| Split regression | PUNCC, crepes, MAPIE | 2x to 5x |
| Normalized / Mondrian split | crepes | 3x / 4x |
| Predictive distributions, CRPS | crepes | 64x to 8,000x; 280x to 760x |
| APS, RAPS, SAPS sets | TorchCP, PUNCC, MAPIE | 2x at 10 classes, 3x to 7x at 100 and 1,000 |
| CQR | MAPIE; PUNCC | 6x; 3x, same intervals |
| CTI | authors' code | 19x, same sets |
| Quantile forest | quantile-forest (Zillow) | fit 4x to 7x; predict 1.2x to 2x |
| MSTL | statsforecast; R forecast | 6x to 7x at equal settings (145x to 520x at defaults); 5x |
| CSP | csp-forecaster 0.2.0 | 2x to 21x |

CIR has no public reference code to compare with. Where a difference in
output exists it is stated in the benchmark: crepes picks one rank too high
in Mondrian groups whose size plus one is a multiple of ten (floating-point
rounding of the rank), which affected 1 group in 20 in our run.

## One scheme for every method

Every method works two ways, with the same code underneath:

```python
from pyconformal import SplitConformalRegressor

# From predictions: any model, any framework, even numbers from a database
cp = SplitConformalRegressor(alpha=0.1)
cp.calibrate(y_pred=pred_cal, y=y_cal)
cp.predict_interval(y_pred=pred_test)

# From a model
cp = SplitConformalRegressor(model, alpha=0.1)
cp.fit(X_train, y_train)           # skip if the model is already trained
cp.calibrate(X_cal, y_cal)
cp.predict_interval(X_test)
```

- `fit` trains the model, `calibrate` uses held-out data, `predict_*` gives
  the output, `update` takes one new observation.
- `alpha` is the miscoverage level (0.1 = 90%). Change it at prediction time
  without recalibrating, or pass a list: `predict_interval(X, alpha=[0.2, 0.1])`.
- Results are plain NumPy arrays.

## Regression: prediction intervals

```python
from sklearn.ensemble import RandomForestRegressor
from pyconformal import SplitConformalRegressor

cp = SplitConformalRegressor(RandomForestRegressor(), alpha=0.1)
cp.fit(X_train, y_train)                  # train the model
cp.calibrate(X_cal, y_cal)                # held-out calibration data
intervals = cp.predict_interval(X_test)   # shape (n, 2): lower, upper
```

Three options, given at `calibrate` and again at `predict_interval`:
`sigma=` (a difficulty estimate per sample; intervals widen where the model
is less sure), `groups=` (a threshold per category, so coverage holds within
each one) and `weights=` (density ratios, for coverage under covariate
shift).

### Without a calibration set: jackknife+ and CV+

```python
from pyconformal import CrossConformalRegressor

cp = CrossConformalRegressor(model, alpha=0.1, cv=10)   # cv="loo" for jackknife+
cp.fit(X, y)                       # one model per fold; every row trains and calibrates
cp.predict_interval(X_test)
```

Prediction costs a few binary searches per sample instead of a pass over the
training set: 17 to 27 times faster than MAPIE on 2,000 to 5,000 training
rows, and it runs on 100,000 rows, where MAPIE builds a training-by-test
matrix and ran out of memory at 20,000 by 20,000 on a 7 GB machine
(`benchmarks/speed_cross.py`).

## Classification: prediction sets

```python
from sklearn.linear_model import LogisticRegression
from pyconformal import SplitConformalClassifier

cp = SplitConformalClassifier(LogisticRegression(), alpha=0.1, score="aps")
cp.fit(X_train, y_train)
cp.calibrate(X_cal, y_cal)
sets = cp.predict_set(X_test)      # boolean (n, n_classes), columns = cp.classes_
```

From probabilities: `calibrate(proba=proba_cal, y=y_cal)`, then
`predict_set(proba=proba_test)`.

Choose the score with `score="lac"`, `"aps"`, `"raps"` or `"saps"`, or pass an
object such as `RAPS(penalty=0.1, k_reg=1)`. APS, RAPS and SAPS are randomised
by default, which gives coverage at the target and smaller sets. Set
`class_conditional=True` for coverage within every class.

## Probability calibration: Venn-Abers

```python
from pyconformal import VennAbers

va = VennAbers(classifier)                # binary or multiclass
va.fit(X_train, y_train)
va.calibrate(X_cal, y_cal)
proba = va.predict_proba(X_test)          # calibrated probabilities
p0_p1 = va.predict_p0_p1(X_test)          # binary: the Venn-Abers pair per sample
```

`CrossVennAbers(classifier, cv=5).fit(X, y)` uses every row for both training
and calibration (binary problems).

From model outputs: `VennAbers().calibrate(proba=proba_cal, y=y_cal)`, or
`scores=` for one score per sample. Install `pyconformal[fast]` to use the
compiled kernel (numba); results are identical with or without it.

## Regression: full predictive distributions

```python
from pyconformal import ConformalPredictiveSystem

cps = ConformalPredictiveSystem(model).calibrate(X_cal, y_cal)
cps.predict_interval(X_test, alpha=0.1)          # (n, 2) intervals
cps.predict_quantile(X_test, [0.1, 0.5, 0.9])    # any quantiles
cps.predict_cdf(X_test, 100.0)                   # P(y <= 100) per sample
cps.crps(X_test, y_test)                         # CRPS per sample, exact
```

Pass `sigma=` (a difficulty estimate per sample) for distributions that widen
on hard samples, and `groups=` to calibrate each category separately; give
them at `calibrate` and again at prediction.

## Regression on quantile predictions: CQR, CTI and CIR

```python
from pyconformal import (
    ConformalizedQuantileRegressor,     # CQR: shift a lower and an upper quantile
    ConformalThresholdedIntervals,      # CTI: keep the short interquantile intervals
    ConformalInterquantileRegressor,    # CIR: grow one interval from the mode
)

cqr = ConformalizedQuantileRegressor(alpha=0.1)
cqr.calibrate(quantiles=q_cal, y=y_cal)          # q_cal: (n, 2) lower, upper
cqr.predict_interval(quantiles=q_test)

cir = ConformalInterquantileRegressor(alpha=0.1, plus=True)
cir.calibrate(quantiles=Q_cal, y=y_cal)          # Q_cal: (n, K + 1) quantiles
cir.predict_interval(quantiles=Q_test)

cti = ConformalThresholdedIntervals(alpha=0.1).calibrate(quantiles=Q_cal, y=y_cal)
cti.predict_set(quantiles=Q_test)                # which intervals are kept
cti.predict_size(quantiles=Q_test)               # total length of the set
```

Or pass a quantile model (one with `predict(X, quantiles=[...])`, or a pair of
lower and upper models for CQR) and use `fit`, `calibrate(X_cal, y_cal)` and
`predict_interval(X_test)`.

CQR intervals are equal-tailed, so they waste width on skewed outcomes. CIR
follows the skew with one interval; CTI can also split into several intervals
for outcomes with more than one mode. On a skewed example
(`benchmarks/speed_quantile.py`, 90% target, all at 90% coverage): mean width
5.59 for CQR, 5.18 for CIR, 5.02 for CIR+ and 4.95 for CTI.

Speed on the same quantile predictions: CQR in 3 ms for 100,000 calibration
and 1,000,000 test points (PUNCC 10 ms, MAPIE 20 ms, identical intervals);
CTI 19 times faster than its authors' code, identical sets.

## Quantile forest

```python
from pyconformal import QuantileForest, ConformalInterquantileRegressor

forest = QuantileForest(n_estimators=100, random_state=0)
cir = ConformalInterquantileRegressor(forest, alpha=0.1, plus=True)
cir.fit(X_train, y_train)          # grows the forest
cir.calibrate(X_cal, y_cal)
cir.predict_interval(X_test)

forest.predict(X_test, quantiles=[0.05, 0.5, 0.95])   # or use it on its own
```

A quantile regression forest (Meinshausen, 2006) written for speed: features
are binned once, splits are found on histograms, trees grow in parallel, and
any number of quantile levels costs one pass per sample. Needs numba
(`pip install pyconformal[fast]`). Against quantile-forest 1.5.0 with the same
settings (`benchmarks/speed_forest.py`, 100,000 training rows, 2 cores): fit
in 3 s against 17 to 21 s; prediction about twice as fast as its exact mode,
with the same accuracy. `max_samples_leaf=1` gives the approximation that is
quantile-forest's default (one stored sample per leaf); in that mode too
pyconformal predicts faster (0.25 s against 0.29 s for 3 levels, 0.31 s
against 0.48 s for 99).

`cir.calibrate_oob(y_train)` calibrates on the forest's out-of-bag
predictions, so no data is set aside. In our runs CQR with out-of-bag
calibration covered 91.1% at a 90% target with intervals 2% narrower than a
half-and-half split; unlike a separate calibration set, this is not a
finite-sample guarantee.

## Time series: STL and MSTL decomposition

```python
from pyconformal import mstl

parts = mstl(y, periods=[24, 168])     # hourly data: day and week
parts.trend, parts.seasonal[24], parts.seasonal[168], parts.remainder
```

Our own implementation of STL (Cleveland et al., 1990) and MSTL (Bandara,
Hyndman and Bergmeir, 2021), with the defaults of R's `forecast::mstl`. It
returns the same numbers as R (to 14 decimal places), and the same numbers as
statsforecast when run with statsforecast's settings
(`jumps=False, inner_iter=5`). Timings from `benchmarks/speed_mstl.py`:

| Series | pyconformal | statsforecast 2.1.1 |
|---|---|---|
| 15-minute data, 6 weeks (4,032 points) | 6 ms | 3,145 ms |
| Hourly data, 1 year (8,760 points) | 12 ms | 1,698 ms |
| Hourly data, 11 years (100,000 points) | 98 ms | 19,300 ms |

Most of that gap is R's default of evaluating each smoother at every tenth
point of its window, which statsforecast does not use. With identical
settings pyconformal is 6 to 7 times faster than statsforecast, and about 5
times faster than R itself.

## Time series: adaptive conformal inference

```python
from pyconformal import AdaptiveConformalInference

aci = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
result = aci.run(y_pred=forecasts, y=actuals)   # replay a whole history
result.intervals                                # (T, 2), each built from past data only
result.coverage

aci.predict_interval(y_pred=next_forecast)      # then go live, one step at a time
aci.update(y_pred=next_forecast, y=next_actual)
```

Four more online methods take the same calls as ACI:

| Class | Idea | 1,000,000 steps |
|---|---|---|
| `DtACI` | Several step sizes at once, weighted by recent performance; nothing to tune | 0.6 s |
| `AgACI` | Combines the thresholds of 30 ACI experts by online aggregation | 4.4 s |
| `NonexchangeableConformal` (NexCP) | Old scores count less: weight `rho ** age` | 0.3 s |
| `ConformalPID` | Moves the threshold itself up after a miss, down after a cover | 0.09 s |

### Ensembles without a calibration set: EnbPI and SPCI

```python
from pyconformal import EnbPI, SPCI

enb = EnbPI(model, alpha=0.1, n_bootstraps=30).fit(X_train, y_train)
result = enb.run(X_test, y_test)       # intervals, with residuals sliding forward
```

`SPCI` has the same calls and predicts the next error from the last few with
the quantile forest. On errors with autocorrelation 0.9 its intervals were
less than half as wide as EnbPI's (3.4 against 7.5), at 85% coverage against
EnbPI's 89% for a 90% target. Both methods have approximate guarantees only,
and SPCI trades some coverage for the narrower intervals.

Works with any forecaster: it needs only predictions and outcomes. Over any
sequence, with no assumptions on the data, the long-run miss rate stays within
`(max(alpha, 1 - alpha) + gamma) / (gamma * T)` of `alpha`.

## Time series: Conformal Seasonal Pools (training-free forecasting)

```python
from pyconformal import ConformalSeasonalPool

csp = ConformalSeasonalPool(random_state=0).fit(history, seasonal_period=24)
forecast = csp.predict(H=48, alpha=0.05, n_samples=100)
forecast.lower, forecast.upper     # (H,) central 95% interval
forecast.quantiles[0.5]            # (H,) any requested quantile
forecast.samples                   # (H, n_samples) predictive draws
```

CSP blends same-season history with conformal residuals around a seasonal
naive forecast. No training, no parameters to learn
([Manokhin, 2026](https://arxiv.org/abs/2605.03789)). This implementation
builds the same predictive distribution as `csp-forecaster` 0.2.0 and runs 2
to 21 times faster. CSP is a forecasting method built from conformal
residuals; unlike the methods above it carries no finite-sample guarantee.

## The guarantee

If the calibration data and the test point are exchangeable (for example,
i.i.d.), the interval or set contains the true value with probability at least
`1 - alpha`. The guarantee is marginal (on average over test points), not
conditional on each input. It holds for any model and any data distribution.

The calibration data must not be used to train the model. If the calibration
set is too small for the chosen `alpha` (fewer than `1/alpha - 1` points), the
intervals are infinite, which is the only way to keep the guarantee.

## Licence

MIT. Copyright (c) 2026 Valery Manokhin.
