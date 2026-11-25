# conformal-tights 0.5.0: audit

25 November 2026. It replaces the first-pass review of 24 November, which used one synthetic data model and listed six parts as not done. All six are done here.

**Audited:** `conformal-tights` 0.5.0 from https://github.com/superlinear-ai/conformal-tights, commit `d8ab41df3fc877e94f839f4b18947b503e5edea1` (21 March 2026), MIT licence. 882 lines: one method (`ConformalCoherentQuantileRegressor`) and a Darts wrapper (`DartsForecaster`). Every source file was read.
**Environment:** Python 3.13, xgboost 3.4.1, scikit-learn 1.9.1, darts 0.48.1, pyconformal 0.1.4.
**Evidence:** every finding has a script here and its raw output in `output/`.

## What the method does

It fits a point model and an XGBoost quantile model (5 levels) on 70% of the data. On 70% of the remaining calibration rows ("level 1") it fits a linear quantile regression, constrained not to cross, from the 5 XGBoost quantiles to the residual. On the other 30% of the calibration rows ("level 2") it shifts each quantile line by a constant. It does this for absolute residuals and for relative residuals and, for each test point, returns whichever of the two is narrower.

## Summary

With its defaults the package under-covers on every one of eight public datasets at every sample size tested. With 5,000 rows the gap is 1 to 5 points and the intervals are sharp. With 1,000 rows or fewer there is no conformal step at all and the gap is 4 to 12 points. The project page lists "requires only a small number of calibration examples" as a feature and "reliable coverage" as another. The package's own two coverage tests pass if coverage reaches 97% of the level asked for (`tests/test_conformal_quantile_regressor.py` line 42, `tests/test_darts_forecaster.py`).

| # | Fault | Measured |
|---|---|---|
| 1 | **No conformal step below 1,414 training rows.** The level-2 shift is applied only if level 2 holds at least 128 rows. Below that the interval is a quantile regression fitted on the calibration set | 300 rows: 80.6% at a 90% target and 84.8% at 95% (mean of 8 datasets). 1,000 rows: 84.2% and 89.9% (7 datasets). Split conformal on the same rows: 90.0% and 95.6%; 89.9% and 94.9% |
| 2 | **Between 425 and 1,413 rows, 30% of the calibration rows are set aside and never used.** The code comment says they go to level 1 in that case; the condition tests the wrong size | 1,000 rows: 300 calibration rows, 210 used, 90 discarded |
| 3 | **The narrower of two calibrated intervals is returned for each test point.** Each of the two is on target; picking the narrower point by point is not | 5,000 rows, 90% target: 87.2% returned (5 datasets, range 85.3% to 89.1%); absolute residuals alone 90.6%, relative alone 91.3%. At 95%: 93.3% returned; 95.2% and 95.6% alone |
| 4 | **Sample weights are ignored in the level-2 shift** (marked TODO in the code), and the shift has no finite-sample correction | One fifth of rows corrupted and given weight 0: the 90% interval is 3.0 times wider than a fit on the clean rows alone (14.5 against 4.8) and covers 99.7% |
| 5 | **Shifts for quantiles between the lowest and the highest are clipped to almost nothing**, to keep the lines from crossing on the calibration points | 16 fits, 5,000 rows: asked 0.035 to 0.052 residual standard deviations at levels 0.25 to 0.75, applied 0.005 to 0.019; applied shift exactly 0 in 25% to 50% of fits. On real data the 0.25 and 0.75 quantiles were hit 27.7% and 71.7% of the time |
| 6 | **`predict_interval` edits a stored array in place.** A later `predict_quantiles` call with the same two levels returns different numbers | 13 of 20 fits at coverage 0.5 (largest change 0.18 standard deviations of y); 15 of 20 at coverage 0.9 when the same floating-point levels are passed |
| 7 | **Quantiles still cross on new points, and intervals can come back upside down.** The constraint holds on calibration points only | Crossing on 0.15% to 0.26% of test points with 5 levels (up to 0.9% in a fit). Lower end above upper end in up to 1.1% of intervals with 300 rows (mean over 40 trials) |
| 8 | **Darts forecaster under-covers out of sample.** Calibration rows are drawn at random from the lagged series, so they sit between training rows in time | README example (Zurich electricity), one step ahead, 2,184 test hours: 92.0% at 95%, 86.8% at 90%, 74.5% at 80%. README's 5-day path forecast, 18 origins: 85.0% at 95% in the first 24 hours. Synthetic AR(1): 92.9%, 87.3%, 76.3% |
| 9 | `historical_forecasts(predict_likelihood_parameters=True)` raises `ValueError` when quantiles are chosen at prediction time, the way the README sets them | Every call |
| 10 | `ŷ.astype(self.y_dtype_)` discards its result | `_coherent_linear_quantile_regressor.py` line 247; no effect measured |

The issue tracker (15 issues) could not be read from the audit environment, so whether any of this has been reported is not known.

## 1 and 2: small samples (`e4_real.py`, `e5_defaults.py`)

`_conformal_coherent_quantile_regressor.py`, lines 107 to 131 and 242. The calibration set is `min(int(0.3 n), 1440)` rows. Level 2 is 30% of it if `round(0.3 n) >= 128`, otherwise one row. The shift is computed only if level 2 has at least 128 rows; that needs 424 calibration rows, so 1,414 training rows. Checked directly: 1,413 rows give 127 level-2 rows and no shift, 1,414 give 128.

Real data, package defaults (`estimator="auto"`: XGBoost defaults), 40 trials per cell. Each trial draws n rows at random for fitting and up to 5,000 other rows for testing. pyconformal gets the same n rows, split 70/30 with the same calibration size, around the same kind of XGBoost model. Coverage at a 90% target / 95% target:

| Dataset | n | Level-2 rows | conformal-tights | pyconformal split | pyconformal CQR |
|---|---|---|---|---|---|
| diabetes | 300 | 1 | 80.9 / 85.4 | 89.7 / 95.3 | 90.2 / 96.1 |
| ames housing | 300 | 1 | 81.6 / 85.4 | 89.7 / 95.2 | 88.4 / 95.0 |
| | 1,000 | 90 | 85.1 / 90.6 | 89.6 / 94.7 | 90.2 / 94.9 |
| abalone | 300 | 1 | 82.9 / 86.4 | 91.3 / 95.8 | 89.9 / 95.7 |
| | 1,000 | 90 | 85.9 / 90.9 | 89.9 / 94.8 | 90.0 / 95.1 |
| kin8nm | 300 | 1 | 81.6 / 85.3 | 89.9 / 95.8 | 90.5 / 95.7 |
| | 1,000 | 90 | 85.6 / 90.5 | 90.0 / 94.9 | 89.9 / 94.8 |
| | 5,000 | 432 | 88.8 / 94.0 | 90.0 / 95.1 | 90.1 / 94.9 |
| grid stability | 300 | 1 | 78.5 / 85.3 | 89.6 / 95.2 | 89.8 / 95.5 |
| | 1,000 | 90 | 81.8 / 89.0 | 89.7 / 95.1 | 90.1 / 95.3 |
| | 5,000 | 432 | 85.3 / 92.3 | 90.3 / 95.2 | 90.1 / 95.1 |
| bike sharing | 300 | 1 | 79.3 / 83.1 | 90.3 / 95.9 | 90.4 / 96.2 |
| | 1,000 | 90 | 82.4 / 88.5 | 89.9 / 95.0 | 89.9 / 95.1 |
| | 5,000 | 432 | 86.1 / 93.0 | 89.9 / 94.9 | 90.0 / 95.2 |
| california housing | 300 | 1 | 81.2 / 85.0 | 89.9 / 95.7 | 89.8 / 95.5 |
| | 1,000 | 90 | 85.6 / 91.1 | 89.8 / 94.8 | 90.2 / 94.8 |
| | 5,000 | 432 | 89.1 / 94.3 | 89.9 / 95.0 | 89.8 / 95.0 |
| superconductivity | 300 | 1 | 79.0 / 82.9 | 89.6 / 95.6 | 90.3 / 95.9 |
| | 1,000 | 90 | 82.8 / 88.4 | 90.3 / 95.1 | 90.3 / 95.2 |
| | 5,000 | 432 | 86.7 / 93.1 | 89.8 / 95.0 | 90.1 / 95.1 |

The largest standard error of a conformal-tights cell is 0.7 points. conformal-tights was below its target in 80% to 100% of trials in every cell; in 13 of the 20 cells it was below in every trial at the 90% target.

Synthetic data with the package's defaults (`e5_defaults.py`, 40 trials): 82.5% / 85.8% at 300 rows, 86.5% / 91.3% at 1,000, 89.0% / 94.0% at 1,500, 89.7% / 94.6% at 5,000. The first pass, with smaller XGBoost models (`e1_coverage.py`, 100 trials), gave 82.4% / 86.6% at 300 and 86.7% / 91.4% at 1,000.

## 3: the narrower of two (`e4_real.py`)

Lines 304 to 309: for each test point the set of quantiles with the smaller standard deviation is returned. With 5,000 rows, 90% target:

| Dataset | Returned | Absolute residuals only | Relative residuals only |
|---|---|---|---|
| kin8nm | 88.8 | 90.6 | 91.1 |
| grid stability | 85.3 | 90.7 | 91.5 |
| bike sharing | 86.1 | 90.5 | 91.6 |
| california housing | 89.1 | 90.7 | 90.9 |
| superconductivity | 86.7 | 90.3 | 91.5 |

In fairness to the method: its intervals adapt to the input and are sharp. With 5,000 rows its interval score was better than that of split conformal and of CQR built on the same rows on four of the five datasets, at both targets (grid stability is the exception). It buys that with 1 to 5 points of coverage, and does not say so.

## 4: sample weights (`e6_weights.py`)

Line 255: `np.quantile(...)` with the comment `# TODO: Use a weighted quantile`. The package marks scikit-learn's weight-invariance check as an expected failure.

Test: 6,000 rows, one fifth with noise ten times larger and weight 0, the rest weight 1; clean test data; 20 trials.

| Fit | Coverage at 90% | Width at 90% | Coverage at 95% | Width at 95% |
|---|---|---|---|---|
| With the weights | 99.7 | 14.5 | 100.0 | 26.4 |
| On the clean rows only | 89.8 | 4.8 | 94.8 | 6.1 |
| No weights, all rows | 99.4 | 18.2 | 99.8 | 31.0 |

The level-1 regression honours the weights; the level-2 shift then undoes it. Mean shift applied to the two ends of the 90% interval: -4.9 and +6.0. With a weighted quantile it would be 0.02 and -0.01.

## 5: clipped shifts (`e8_clip.py`)

`intercept_clip` in `_coherent_linear_quantile_regressor.py`, applied at line 256. Shifts in units of the standard deviation of the level-2 residuals, absolute-residual branch:

| Runs | Level | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|---|
| Synthetic, 5,000 rows, 16 fits | asked | 0.150 | 0.052 | 0.035 | 0.044 | 0.113 |
| | applied | 0.091 | 0.006 | 0.005 | 0.019 | 0.076 |
| | observed frequency | 0.053 | 0.258 | 0.502 | 0.742 | 0.946 |
| Synthetic, 2,000 rows, 20 fits | asked | 0.231 | 0.091 | 0.047 | 0.069 | 0.135 |
| | applied | 0.163 | 0.006 | 0.006 | 0.003 | 0.068 |
| | observed frequency | 0.054 | 0.257 | 0.500 | 0.738 | 0.946 |
| Four real datasets, 5,000 rows, 12 fits | asked | 0.072 | 0.032 | 0.021 | 0.040 | 0.089 |
| | applied | 0.067 | 0.007 | 0.011 | 0.004 | 0.086 |
| | observed frequency | 0.059 | 0.277 | 0.501 | 0.717 | 0.936 |

In the relative-residual branch the applied shift at levels 0.25 and 0.50 was exactly zero in every one of the 12 real-data fits. The largest gap between a quantile's observed frequency and its level in a single fit was 6.9 points.

## 6: call order (`e2b_call_order.py`)

`predict_interval(coverage=c)` stores its two shifts under the key `((1-c)/2, 1-(1-c)/2)` and then, because it asks for `priority="coverage"`, overwrites entries of the stored array (lines 292 to 297). `predict_quantiles` with the same two numbers then reads the edited array, although its own default is `priority="accuracy"`. With coverage 0.5 the key is `(0.25, 0.75)` exactly, so `predict_quantiles(X, quantiles=(0.25, 0.75))` is affected. With coverage 0.9 the key is `(0.04999999999999999, 0.95)`, so a user who types `0.05` gets a different key and is not affected. The first-pass test typed `0.05` and so reported no effect; that result is withdrawn.

## 8 and 9: Darts forecaster (`e7_darts.py`)

Quantiles are read from 500 sample paths, as in the README.

| Test | 95% | 90% | 80% | 50% |
|---|---|---|---|---|
| Synthetic hourly series, AR(1) errors with coefficient 0.9, one step ahead, 5 series of 1,000 test hours | 92.9 | 87.3 | 76.3 | 47.4 |
| README example, one step ahead, 2,184 test hours | 92.0 | 86.8 | 74.5 | 46.1 |
| README example, 5-day path forecast, 18 origins: hours 1 to 24 | 85.0 | 78.7 | 69.9 | 47.0 |
| hours 25 to 72 | 91.6 | 85.1 | 76.4 | 49.3 |
| hours 73 to 120 | 93.8 | 90.2 | 84.1 | 57.8 |

The README example has 432 level-2 rows, so the shift is applied; the shortfall is not the small-sample fault. The 18 origins cover 90 consecutive days of one summer and are not independent.

## Speed

The non-crossing fit is a linear programme solved the first time a set of levels is requested. First `predict_quantiles` call with 5 levels on 5,000 training rows, one thread: median 51 s on synthetic data, 39 s on real data. For a two-level interval, fit plus prediction took 3 to 11 s at 5,000 rows against 0.1 to 1.3 s for split conformal around the same model.

## Anything pyconformal lacks?

No method. The ideas worth borrowing are conveniences: a Darts wrapper, a no-argument default (`"auto"`), and targets of datetime type.

## What passed

| Part | Result |
|---|---|
| Each branch alone with 5,000 rows | 90.3% to 91.6% at 90%, 95.1% to 95.8% at 95% |
| Median quantile | Observed frequency 0.500 to 0.502 |
| Non-crossing on calibration points | Holds by construction |
| pandas input | DataFrame in, DataFrame out |

## Coverage of this audit

| Part | Status |
|---|---|
| `_conformal_coherent_quantile_regressor.py`, `_coherent_linear_quantile_regressor.py`, `_xgboost_weighted_quantile.py` | Read and tested |
| `_darts_forecaster.py` | Read; tested on one synthetic design and the README example. Categorical covariates, several series at once and `output_chunk_length` above 1 not tested |
| Non-default `nonconformity_quantiles`, `conformal_calibration_size`, `coherence_buffer` | Not tested |
| A pre-fitted `estimator`; datetime and timedelta targets | Not tested |
| Sample weights | One design (zero weights on corrupted rows). Covariate-shift weights not tested |
| Issue tracker | Not read (not reachable from the audit environment) |

## How to reproduce

    pip install conformal-tights==0.5.0 xgboost darts pyconformal
    mkdir data out
    python get_data.py data                          # downloads the eight datasets from scikit-learn and OpenML
    python e4_real.py data california 40 out         # one dataset; then: python e4_report.py out
    python e5_defaults.py 40
    python e6_weights.py 20
    python e7_darts.py
    python e8_clip.py synth5000 16                   # also: synth2000 20; real 3 data
    python e2b_call_order.py

`e1_coverage.py`, `e2_call_order.py` and `e3_bias_clip.py` are the first-pass scripts; `e2_call_order.py` cannot detect fault 6 (see above).
