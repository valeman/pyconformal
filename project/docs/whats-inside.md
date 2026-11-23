# What's inside

Every method in pyconformal. Click a name for a minimal example you can run.

## Regression

| Method | What it gives you | Class or function |
|---|---|---|
| [Split conformal regression](methods/split-regression.md) | Prediction intervals around any regression model. | `SplitConformalRegressor` |
| [Mondrian (group-conditional) conformal prediction](methods/mondrian.md) | One threshold per category, so coverage holds inside every category and not only on average. | `SplitConformalRegressor`, `SplitConformalClassifier`, `ConformalPredictiveSystem` |
| [Normalized (locally adaptive) intervals](methods/normalized.md) | Intervals that are wider where the model is less sure, using a difficulty estimate `sigma`. | `SplitConformalRegressor` |
| [Weighted conformal prediction (covariate shift)](methods/weighted.md) | Intervals that stay valid when the test inputs come from a different distribution than the calibration inputs. | `SplitConformalRegressor` |
| [Jackknife+ and CV+](methods/jackknife-plus.md) | Intervals without setting data aside for calibration. | `CrossConformalRegressor` |

## Regression on quantiles

| Method | What it gives you | Class or function |
|---|---|---|
| [Conformalized quantile regression (CQR)](methods/cqr.md) | Shifts a lower and an upper predicted quantile until they cover. | `ConformalizedQuantileRegressor` |
| [Conformal interquantile regression (CIR, CIR+)](methods/cir.md) | Grows one interval from a grid of predicted quantiles. | `ConformalInterquantileRegressor` |
| [Conformal thresholded intervals (CTI)](methods/cti.md) | Keeps the shortest interquantile intervals; the result can be a union of intervals. | `ConformalThresholdedIntervals` |
| [Quantile forest](methods/quantile-forest.md) | A fast quantile regression forest, usable on its own or under CQR, CIR and CTI. | `QuantileForest` |
| [Quantile models from CatBoost, XGBoost and LightGBM](methods/quantile-booster.md) | One wrapper that makes any of the three libraries predict a grid of quantiles. | `QuantileBooster` |

## Classification

| Method | What it gives you | Class or function |
|---|---|---|
| [Mondrian classification: by category or by class](methods/mondrian-classification.md) | Prediction sets with one threshold per category (`groups=`) or per class (`class_conditional=True`). | `SplitConformalClassifier` |
| [Split conformal classification (LAC, APS, RAPS, SAPS)](methods/classification.md) | Prediction sets that contain the true class with the chosen probability. | `SplitConformalClassifier`, `LAC`, `APS`, `RAPS`, `SAPS` |

## Probability calibration

| Method | What it gives you | Class or function |
|---|---|---|
| [Venn-Abers calibration](methods/venn-abers.md) | Calibrated probabilities from any classifier's scores, binary or multi-class. | `VennAbers`, `CrossVennAbers` |

## Predictive distributions

| Method | What it gives you | Class or function |
|---|---|---|
| [Conformal predictive systems](methods/predictive-systems.md) | A whole calibrated predictive distribution for each prediction, not one interval. | `ConformalPredictiveSystem` |
| [When the outcome is a whole distribution](methods/distributions.md) | Conformal sets for vector-valued predictions, such as the coefficients of a predicted distribution. | `ConformalDistributionRegressor`, `cosine_coefficients`, `cosine_cdf`, `cosine_density` |

## Time series

| Method | What it gives you | Class or function |
|---|---|---|
| [Adaptive conformal inference (ACI)](methods/aci.md) | Intervals that widen and narrow online so the long-run miss rate stays on target, whatever the data does. | `AdaptiveConformalInference` |
| [DtACI, AgACI, Conformal PID and NexCP](methods/dtaci-agaci-pid-nexcp.md) | Four more online methods with the same interface as ACI. | `DtACI`, `AgACI`, `ConformalPID`, `NonexchangeableConformal` |
| [Multi-step forecasts: MSCP and AcMCP](methods/multi-step.md) | Intervals for forecasts several steps ahead, one calibration per horizon. | `MultiStepSplitConformal`, `AcMCP` |
| [EnbPI and SPCI](methods/enbpi-spci.md) | Bootstrap ensembles that give intervals without a calibration set, with residuals sliding forward. | `EnbPI`, `SPCI` |
| [Conformal Seasonal Pools](methods/csp.md) | A training-free probabilistic forecaster for seasonal series. | `ConformalSeasonalPool` |
| [STL and MSTL decomposition](methods/stl-mstl.md) | Fast trend and seasonal decomposition, for one or several seasonal periods. | `stl`, `mstl` |

## Building blocks

| Method | What it gives you | Class or function |
|---|---|---|
| [The conformal quantile, on its own](methods/conformal-quantile.md) | The exact split-conformal threshold for a set of scores, in exact arithmetic. | `conformal_quantile` |

## Find it by the name you know

| You are looking for | It is here |
|---|---|
| Mondrian, group-conditional, category-wise coverage | [Mondrian regression](methods/mondrian.md), [Mondrian classification](methods/mondrian-classification.md) |
| Class-conditional, label-conditional | [Mondrian classification](methods/mondrian-classification.md) (`class_conditional=True`) |
| Inductive conformal prediction (ICP) | [Split conformal regression](methods/split-regression.md), [classification](methods/classification.md) |
| Locally adaptive, normalized, difficulty estimate | [Normalized intervals](methods/normalized.md) |
| Covariate shift, importance weights | [Weighted conformal prediction](methods/weighted.md) |
| CV+, jackknife+, cross-conformal | [Jackknife+ and CV+](methods/jackknife-plus.md) |
| CQR, quantile regression | [CQR](methods/cqr.md), [CIR](methods/cir.md), [CTI](methods/cti.md) |
| CatBoost, XGBoost, LightGBM | [Quantile models from boosting libraries](methods/quantile-booster.md) |
| Probability calibration, Platt or isotonic replacement | [Venn-Abers](methods/venn-abers.md) |
| Predictive distribution, CDF, CRPS | [Conformal predictive systems](methods/predictive-systems.md) |
| Online, distribution shift, drift | [ACI](methods/aci.md), [DtACI, AgACI, PID, NexCP](methods/dtaci-agaci-pid-nexcp.md) |
| Forecast horizons | [Multi-step forecasts](methods/multi-step.md) |
| APS, RAPS, SAPS, LAC, THR | [Classification scores](methods/classification.md) |

## Not in the library yet

Conformal risk control, conformal object detection, graph and language-model methods, and full (transductive) conformal prediction. Ask for one by opening an [issue](https://github.com/valeman/pyconformal/issues).
