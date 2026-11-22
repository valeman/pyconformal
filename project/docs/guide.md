# pyconformal user guide

One section per family of methods, each with a short example.

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

The choice of window matters when the outcome has more than one mode. On a
two-mode outcome (`benchmarks/review_experiments.py cir`) the authors' rule
covered 88.3% at a 90% target and 76.7% at an 80% target, in their own code
and in ours with `window="shortest"`; the default covered 90.5% and 80.4%.
The authors' rule has no coverage guarantee at all: with fixed predicted
quantiles `[0, 1, 11, 13, 15]` and every outcome equal to 2 (exchangeable
data, 90% target), the default interval is `[0, 11]` and covers every
outcome, while `window="shortest"` returns `[11, 15]` and covers none.
`window="shortest"` is kept only to reproduce the authors' code, and warns
when used.
For outcomes like that, CTI is the better tool: it may return two intervals,
and its sets were 30% to 40% shorter than any single interval.

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
(`pip install "pyconformal[fast]"`). Against quantile-forest 1.5.0 with the same
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

## Quantile models from CatBoost, XGBoost and LightGBM

```python
from pyconformal import QuantileBooster, ConformalInterquantileRegressor

qb = QuantileBooster("catboost", levels=19, iterations=300)   # or "xgboost", "lightgbm"
cir = ConformalInterquantileRegressor(qb, plus=True, alpha=0.1)
cir.fit(X_train, y_train).calibrate(X_cal, y_cal)
cir.predict_interval(X_test)
```

`QuantileBooster` trains `levels` quantile levels (19 gives 0.05 to 0.95) with
the named library and answers `predict(X, quantiles=...)` for any levels, which
is what CQR, CTI and CIR ask of a model. CatBoost and XGBoost train one model
for all levels; LightGBM trains one per level. The trained quantiles are
sorted so they never cross, and levels in between are interpolated. Levels 0
and 1 are set a fixed distance beyond the lowest and highest trained level,
measured on a tenth of the training rows kept out of the boosting
(`tail_holdout`), so that about 1% of new outcomes fall beyond each end.
Install the library you use: `pip install "pyconformal[catboost]"`.

Compared with Zillow's quantile-forest and our own forest
(`benchmarks/quantile_models.py`; 5 random splits per dataset, half the rows
for training, a quarter each for calibration and test, 90% target, 100 trees
for the forests, 300 rounds for the boosters, 2 cores). Pinball loss is of
the raw quantiles; widths are after the conformal step. Coverage after
calibration was between 89.4% and 90.5% in every row for CQR and CIR+.

| Dataset | Model | Pinball | CQR width | CIR+ width | Fit | Predict |
|---|---|---|---|---|---|---|
| California housing | quantile-forest (Zillow) | 0.129 | 1.87 | 1.43 | 1.5 s | 0.14 s |
| | `QuantileForest` | 0.124 | 1.70 | 1.41 | 0.23 s | 0.08 s |
| | CatBoost | 0.119 | 1.36 | 1.38 | 6.4 s | 0.02 s |
| | XGBoost | 0.123 | 1.49 | 1.40 | 9.2 s | 0.28 s |
| | LightGBM | 0.116 | 1.36 | 1.34 | 7.1 s | 0.98 s |
| kin8nm | quantile-forest (Zillow) | 0.048 | 0.573 | 0.532 | 0.65 s | 0.07 s |
| | `QuantileForest` | 0.045 | 0.541 | 0.513 | 0.11 s | 0.03 s |
| | CatBoost | 0.035 | 0.427 | 0.459 | 4.2 s | 0.01 s |
| | XGBoost | 0.045 | 0.529 | 0.557 | 7.0 s | 0.11 s |
| | LightGBM | 0.038 | 0.529 | 0.527 | 5.2 s | 0.47 s |
| Synthetic, skewed | quantile-forest (Zillow) | 0.455 | 5.02 | 4.70 | 1.5 s | 0.14 s |
| | `QuantileForest` | 0.454 | 4.98 | 4.62 | 0.20 s | 0.08 s |
| | CatBoost | 0.452 | 4.77 | 4.68 | 5.5 s | 0.02 s |
| | XGBoost | 0.453 | 4.68 | 4.48 | 8.0 s | 0.28 s |
| | LightGBM | 0.454 | 4.56 | 4.59 | 6.5 s | 0.78 s |
| Synthetic, two modes | quantile-forest (Zillow) | 0.824 | 6.74 | 7.61 | 1.8 s | 0.14 s |
| | `QuantileForest` | 0.821 | 6.67 | 8.08 | 0.21 s | 0.08 s |
| | CatBoost | 0.825 | 6.58 | 7.38 | 5.5 s | 0.02 s |
| | XGBoost | 0.813 | 6.50 | 6.71 | 7.7 s | 0.27 s |
| | LightGBM | 0.825 | 6.44 | 6.59 | 6.3 s | 0.74 s |

What to take from it:

- **Against Zillow's forest, ours** has equal or lower pinball loss on all
  four datasets, fits 6 to 8 times faster and predicts about twice as fast.
  On the two-mode data its CIR+ intervals were 6% wider.
- **Boosters** gave the shortest CQR intervals on every dataset, by up to 27%
  on California housing, and take 20 to 40 times longer to train than our
  forest. Their raw 5%-95% intervals cover only 71% to 87%, so they need the
  conformal step; the forests' raw intervals cover 91% to 97%.
- **Plain CIR over-covers with boosters** (91% to 93%), because many rows tie
  at the same number of steps. Use `plus=True`.
- **On small data, use the forest.** On diabetes (221 training rows) the
  boosters' pinball loss was 11% to 20% worse than the forests', and CIR
  returned infinite intervals for all three, because too many calibration
  outcomes fell outside their predicted range. CQR was fine (width 186 to 191
  against 186 and 194 for the forests).

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

The reported `coverage` is the share of outcomes inside the issued
intervals. When the working level asks for the empty set, the interval is
returned as `(nan, nan)` and counts as a miss.

Four more online methods take the same calls as ACI:

| Class | Idea | 1,000,000 steps |
|---|---|---|
| `DtACI` | Several step sizes at once, weighted by recent performance; nothing to tune | 0.6 s |
| `AgACI` | Combines the thresholds of 30 ACI experts by online aggregation | 4.4 s |
| `NonexchangeableConformal` (NexCP) | Old scores count less: weight `rho ** age` | 0.3 s |
| `ConformalPID` | Moves the threshold itself up after a miss, down after a cover. The step is proportional to the range of recent scores; when they are all equal (a constant series) a nonzero step is kept so the threshold cannot get stuck | 0.09 s |

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

Every residual comes from members that did not see that row. If some rows
are in every bootstrap sample (likely with few members), `fit` trains extra
members until each row has been left out at least once; `n_members_` gives
the final number. With few members and in-bag residuals instead, coverage at
a 90% target fell to 84% in the audit's test; with out-of-bag residuals only
it is 90%.

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
