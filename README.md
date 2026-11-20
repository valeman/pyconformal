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

- **Venn-Abers:** the same answers as pyconformal, 5 to 128 times slower
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
for binary and multiclass problems, up to 128 times faster than existing
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
and SAPS scores, class-conditional sets), conformal
predictive distributions, Venn-Abers calibration (binary and multiclass),
adaptive conformal inference (ACI) and conformal seasonal pools (CSP) for time
series. Next: conformalized quantile regression with a fast quantile forest.

## Install

```bash
pip install pyconformal
```

## Regression: prediction intervals

```python
from sklearn.ensemble import RandomForestRegressor
from pyconformal import SplitConformalRegressor

model = SplitConformalRegressor(RandomForestRegressor(), alpha=0.1)
model.fit(X_train, y_train)        # train the model
model.calibrate(X_cal, y_cal)      # held-out calibration data
intervals = model.predict_interval(X_test)   # shape (n, 2): lower, upper
```

## Classification: prediction sets

```python
from sklearn.linear_model import LogisticRegression
from pyconformal import SplitConformalClassifier

model = SplitConformalClassifier(LogisticRegression(), alpha=0.1)
model.fit(X_train, y_train)
model.calibrate(X_cal, y_cal)
sets = model.predict_set(X_test)   # boolean (n, n_classes), columns = model.classes_
```

Choose the score with `score="lac"`, `"aps"`, `"raps"` or `"saps"`, or pass an
object such as `RAPS(penalty=0.1, k_reg=1)`. APS, RAPS and SAPS are randomised
by default, which gives coverage at the target and smaller sets. Set
`class_conditional=True` for coverage within every class.

Already have a trained model? Pass `prefit=True` and skip `fit`.

## Probability calibration: Venn-Abers

```python
from pyconformal import VennAbersCalibrator

model = VennAbersCalibrator(classifier)      # binary or multiclass
model.fit(X_train, y_train)
model.calibrate(X_cal, y_cal)
proba = model.predict_proba(X_test)          # calibrated probabilities
p0_p1 = model.predict_p0_p1(X_test)          # binary: the Venn-Abers pair per sample
```

Works on raw scores too: `VennAbers().fit(scores_cal, y_cal).predict_proba(scores_test)`.
Install `pyconformal[fast]` to use the compiled kernel (numba); results are
identical with or without it.

## Regression: full predictive distributions

```python
from pyconformal import ConformalPredictiveSystem

cps = ConformalPredictiveSystem().fit(y_cal, model.predict(X_cal))
y_pred = model.predict(X_test)
cps.predict_interval(y_pred, alpha=0.1)          # (n, 2) intervals
cps.predict_quantile(y_pred, [0.1, 0.5, 0.9])    # any quantiles
cps.predict_cdf(y_pred, 100.0)                   # P(y <= 100) per sample
cps.crps(y_pred, y_test)                         # CRPS per sample, exact
```

Pass `sigma_cal` / `sigma` (a difficulty estimate per sample) for
distributions that widen on hard samples, and `groups_cal` / `groups` to
calibrate each category separately.

## Time series: adaptive conformal inference

```python
from pyconformal import AdaptiveConformalInference

aci = AdaptiveConformalInference(alpha=0.1, gamma=0.01)
result = aci.run_intervals(y_pred, y_true)   # replay a whole history
result.intervals                             # (T, 2), each built from past data only
result.coverage

aci.predict_interval(next_forecast)          # then go live, one step at a time
aci.update(abs(next_actual - next_forecast))
```

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
