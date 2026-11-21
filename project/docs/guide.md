# pyconformal user guide

One section per family of methods, each with a short example. Every speed figure names the script in [`benchmarks/`](../benchmarks/) that produces it.

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

Against the CIR authors' code (`benchmarks/speed_cir.py`): with
`window="shortest"` pyconformal returns the same interval as their code on
every one of 200,000 test points, in 0.12 s against 1.85 s. The default uses
the same growing window in calibration and prediction, which is what the
coverage proof needs; their code scores calibration points with the growing
window and predicts with the shortest window of that many intervals. For
CIR+ their code covered 89.90% at a 90% target over 3,000 repetitions
(standard error 0.03); pyconformal's covered 90.00%.

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
settings (`benchmarks/speed_forest.py`), on a 16-core server with 100,000 and
1,000,000 training rows:

| | pyconformal | quantile-forest |
|---|---|---|
| Fit | 1.9 s and 27 s | 11 to 16 s and 215 to 258 s (6 to 9 times slower) |
| Predict 20,000 rows, exact | 0.41 s and 0.62 s | 1.15 s and 1.89 s |
| Predict, one sample per leaf | 0.14 s and 0.40 s | 0.25 s and 0.66 s |
| Accuracy (pinball loss) | same to the third decimal | |

`max_samples_leaf=1` is the approximation that quantile-forest uses by
default (one stored sample per leaf). The server was busy during the test, so
read the ratios, not the seconds. Prediction has since been made about a third
faster on one thread; that change has not been re-measured on the server.

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

### Several steps ahead: multi-step split conformal prediction

```python
from pyconformal import MultiStepSplitConformal

ms = MultiStepSplitConformal(alpha=0.1)
result = ms.run(y_pred=F, y=y)            # F[t, h - 1]: forecast for time t made h steps earlier
result.intervals                           # (T, H, 2), one calibration per horizon
ms.predict_interval(y_pred=next_forecasts) # the H steps after the history
```

Each horizon has its own errors, and the interval for a target uses only the
errors that were known when its forecast was made. Same results as `scp` in
the R package conformalForecast, in about 10 ms against 42 s for 3,000 time steps and
12 horizons (`benchmarks/speed_multistep.py`). At a few calibration sizes the
R package takes one rank higher than the exact rule (floating-point rounding);
pyconformal computes the rank exactly.

### Several steps ahead, online: AcMCP

```python
from pyconformal import AcMCP

ac = AcMCP(alpha=0.1, min_calibration=50)
result = ac.run(y_pred=F, y=y)             # same inputs as MultiStepSplitConformal
ac.predict_interval(y_pred=next_forecasts)
```

Autocorrelated multi-step conformal prediction (Wang and Hyndman) tracks the
two ends of the interval at each horizon from past misses, adds an integrator,
and forecasts the next error with an MA(h - 1) model and a regression on the
shorter horizons. Without the scorecaster the results equal `acmcp` in the R
package conformalForecast to 14 digits, in 0.8 ms against 3.9 s (1,000 time
steps, 6 horizons). With the scorecaster: 2.5 s against 107 s; the median
difference from R is 0.00001 and the largest 0.01, because both fit the MA
model with an iterative optimiser and R's stops earlier, sometimes at a worse
fit. `refit_every=10` refits the MA model every tenth step and is about 8
times faster again.

Two things differ from the R defaults on purpose. `KI=None` scales the
integrator by the largest error seen so far; R's default uses the largest
error of the whole history, which is not known at the time. Pass a number to
reproduce R. The scorecaster needs numba to run at a useful speed.

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
