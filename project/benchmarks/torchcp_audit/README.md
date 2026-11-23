# TorchCP 1.2.1: forensic audit

23 November 2026. Package `torchcp` 1.2.1 from PyPI (uploaded 14 October 2025), git tag `v1.2.1`, commit `3541cde18b2447cd54e7ca6a814eb0d60585c3f9`. All 89 Python files in the wheel are byte-identical to the tag. Every fault below is also in `master` at `627455f` (20 July 2026). Receipts: `CITATIONS.md`.

Method: fixed models or log-probabilities passed straight in, so any coverage gap comes from the conformal step and not from training. Scripts are in this folder, raw output in `output/`.

**Status: every public part has now been run or read; the few limits are listed at the end.** Nothing here has been reported to the maintainers by us. Their tracker (22 issues, 8 open, read 23 November 2026) already holds fault 8 as issue 126; none of the others is there.

## Faults

| # | Fault | Where | Measured |
|---|---|---|---|
| 1 | **CQR-R intervals are not conformalised.** The width used to scale the correction is read from the freshly zeroed output tensor, not from the predicted quantiles, so the correction is q_hat x 0.000001. The interval returned is the raw quantile band. | `regression/score/cqrr.py` line 39 (compare line 29) | Raw band covers 57.6%; `CQRR` returns 57.6% at a 90% target. `CQR`, `CQRM`, `CQRFM` on the same data: 90.1% to 90.2%. (`r1_regression.py`) |
| 2 | **ACI is not Gibbs and Candès' method: the level is clipped.** alpha_t is held inside [1/(n+1), 0.9999], so the interval can never become infinite and the long-run guarantee is lost. | `regression/predictor/aci.py` line 197 | gamma 0.05, 3,000 steps, bound on the miss rate 10.6%. Random-walk error: 14.0% (worst run 15.1%); trending series with a fixed model: 11.0%; noise growing 1% a step: 11.0%. Unclipped reference on the same sequences: 9.9% to 10.1%. TorchCP gave no infinite interval in any run; the reference needed them on 14% to 39% of steps. (`r2_aci.py`) |
| 3 | **AgACI is neither ACI nor an aggregation.** (a) Each expert's level is `alpha + gamma (alpha - window error rate)`: nothing is carried from step to step. (b) The weights are computed from the same past intervals for every expert, so all experts get the same weight at every step; the output is their plain average. (c) Infinite bounds are replaced by -99999 and 99999. | `regression/predictor/agaci.py` lines 88, 104 to 113, constructor default `threshold` | Miss rate at a 10% target: 44.5% (trending series), 38.2% (noise growing), 27.5% (random-walk error), 11.1% (one jump in scale), 10.0% (i.i.d.). Identical weights confirmed at every step of every run. (`r2_aci.py`) |
| 4 | **NorABS cannot predict more than one row.** A shape error in the interval formula: any batch of 2 or more rows raises `RuntimeError`. | `regression/score/norabs.py` lines 286 to 289 | Batch of 1 works; 2 and 64 fail, for `variance` and `knn_distance`. (`q2_more.py` part c0) |
| 5 | **NorABS docstring leads to under-coverage.** The argument `data_loader` is documented as "Calibration dataset". Neighbours are then searched in the calibration set itself, so each calibration row is its own nearest neighbour. TorchCP's own example passes the training set instead, which is correct. | `regression/score/norabs.py` lines 224 and 234 | Calibration set as documented: 83.6% (`knn_residual`, k = 5), 84.6% (`knn_label`, k = 5), 88.7% to 88.9% (k = 20), 97.1% (`knn_distance`). Separate set: 90.2% to 90.6%. Same mechanism as crepes fault 1. (`q2_more.py` part c) |
| 6 | **ClusteredPredictor returns every label for classes without a cluster.** The threshold for the "null" cluster is infinity; the authors' code (Ding et al.) uses the ordinary split threshold there, and that line is present but commented out in TorchCP. Valid, but useless where it applies. | `classification/predictor/cluster.py` lines 323 to 324 | 4 classes and 10 classes: every set contains every label (coverage 100%). 100 classes with 30 rare: mean set size 34.3 against 6.0 for plain split, same coverage. 100 balanced classes: fine (6.2). (`q2_more.py` part a) |
| 7 | **`SAPS(randomized=False)` returns empty sets on accurate models.** The top class scores 0 in calibration and p_max in prediction. Once more than 90% of calibration labels are the top class, the threshold is 0. Default (`randomized=True`) is fine. | `classification/score/saps.py` line 62 | Top-1 accuracy 0.93 and 0.98: coverage 0.0% at a 90% target. Accuracy 0.88 and below: on target. APS, RAPS and TOPK without randomisation are fine. (`q3_nonrandom.py`) |
| 8 | **`WeightedPredictor` is not weighted conformal prediction.** Calibration weights are sorted on their own, separately from the scores, so the largest weights land on the largest scores. Already reported by another user (issue 126, open). Also, `alpha` given to the constructor is overwritten with `None`. | `classification/predictor/weight.py` lines 95, 170 and 49 | Covariate shift with the true weights supplied: 98.2% and 99.2% at a 90% target, sets 41% to 62% larger than the correct weighted method (90.0%, 90.4%). Over-covers, so valid but wasteful. (`t3_rest.py` part a) |
| 9 | **`ConformalLM` with a scaler loses its risk guarantee.** Scores are scaled once before tuning and calibration and then again inside `predict_with_config`; at test time they are scaled once. Thresholds are calibrated on one scale and applied on another. | `llm/predictor/conformal_llm.py` lines 120, 181 and 295 | Platt scaling, allowed failure rate 5%: true risk above the limit in 25.5% of calibration draws (no rejection) and in 96% to 100% (with rejection); mean risk 0.367 against a limit of 0.30. Without scaling: 0% to 5%, as it should be. Synthetic scores, no language model. (`t3_rest.py` part f) |
| 10 | **`CDLoss` (C-Adapter) has the wrong sign.** Minimising it pushes wrong-label scores below the true-label score, the opposite of the paper's stated aim. | `classification/loss/cd.py` line 67 | LAC, 6 classes: set size 3.38 before, 5.83 after minimising it; with the sign flipped, the gap between wrong and true labels widens as intended. With non-randomised APS neither direction helped. Coverage is unaffected (split calibration follows). Medium confidence on the paper reading: equation not checked line by line. (`t3_rest.py` part b) |
| 11 | **LLM metric `SSCL` is not a loss.** It sums set membership over examples instead of over positions, so "set sizes" are per position. | `llm/utils/metrics.py` line 69 | Returns 101.0 on a 300 x 8 example where a mean loss must lie in [0, 1]. (`t3_rest.py` part c2) |

Smaller:

| Item | Detail |
|---|---|
| `RC3PPredictor.calculate_threshold` | Fails with `TypeError` when called directly, because `num_classes` is only set inside `calibrate`. |
| `Margin(score_type="identity")` | Overwrites the caller's probability tensor (true-label entries set to -1). `classification/score/margin.py` line 43. |
| Models without parameters | Any predictor built on a parameter-free module raises `StopIteration` (`utils/common.py`, `get_device`). |
| ACI miss indicator | A value exactly on the bound counts as a miss (`>=`, `<=`). |
| `ACIPredictor.train` | Calibrates on the training data itself. |
| `SCPOTrainer` | Needs a model with an attribute `fc`; any other model raises `AttributeError`. |
| `ConformalPredictiveDistribution.predict` | Values are returned unsorted. |

## What passed

| Part | Result |
|---|---|
| Rank in `calculate_conformal_value` | Exact in 9,835 of 9,835 cases (five levels, n up to 2,000). |
| Classification scores with `SplitPredictor` (LAC, APS, APS not randomised, RAPS, SAPS default, Margin, TOPK, EntmaxScore) | 89.7% to 89.9% at 90%; same with one row per call (no batch-size effect). TOPK not randomised 92.0% (ties). `SAPS(randomized=False)`: see fault 7. |
| Small calibration sets (9 to 99 rows) | On target; infinite threshold when no rank exists. |
| `ClassConditionalPredictor` | 89.6% to 90.9% inside each of four classes. |
| `RC3PPredictor` (through `calibrate` path, `num_classes` set) | 90.3% to 93.2% inside each class. |
| p-values, classification and regression | Uniform: P(p <= 0.1) = 0.100 to 0.103. |
| KNN score | 90.7%. |
| Regression `ABS`, `CQR`, `CQRM`, `CQRFM` | 90.1% to 90.2%. |
| `EnsemblePredictor` (EnbPI), i.i.d. data | 90.1%. |
| Graph: APS, DAPS, SNAPS with `SplitPredictor`; `NAPSPredictor` | 90.1% to 90.7%. |

| `R2CCP`, trained | 89.8%. |
| `ConformalPredictiveDistribution` | Quantiles calibrated: 9.9%, 48.7%, 89.2% at 10%, 50%, 90%. |
| `ConformalLM` without scaling | Risk above the limit in 0% to 5% of draws at an allowed 5%. |
| Metrics: `coverage_rate` (both types), `average_size`, `CovGap`, `VioClasses`, `SSCV`, `singleton_hit_ratio`, regression `coverage_rate` and `average_size` | Equal to an independent calculation. |
| Trainers followed by split calibration (`TSTrainer`, `ConfTSTrainer`, `ConfTrTrainer`, `UncertaintyAwareTrainer`, `OrdinalTrainer`) | Coverage 89.2% to 89.8%. None made sets smaller than plain cross-entropy on this small problem (3.53 to 3.57 against 3.58); that is not a fault. |
| `ACIPredictor.train` and `.evaluate` end to end, with retraining | Miss rate 10.0% to 11.1% on three AR(1) series, inside the bound. The clipping in fault 2 needs stronger drift to show. |
| `EnsemblePredictor.evaluate` on dependent data | 89.5% stable; 82.3% after a fourfold jump in noise; 58.3% with noise growing 0.5% a step. A plain sliding window of the same length gives 90.2%, 85.0%, 65.1%: this is the method, not a coding fault. |

## Limits

| Part | State |
|---|---|
| `ConformalLM` | Tested on synthetic scores only; text generation and scoring with a real language model not run. |
| `WSC` and the p-value criteria metrics | Not checked. Another user reports swapped arguments in `WSC` (issue 127, open). |
| Other open LLM reports by other users (issues 124 and 128) | Not re-checked. |
| `CFGNNTrainer`, `SCPOTrainer` | Not run (the second fails on models without `fc`). |
| Loss formulas | Compared with the papers only for `CDLoss`; the others were run, not derived. |
| GPU | CPU only. |
