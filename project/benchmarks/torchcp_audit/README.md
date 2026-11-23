# TorchCP 1.2.1: forensic audit

23 November 2026. Package `torchcp` 1.2.1 from PyPI (uploaded 14 October 2025), git tag `v1.2.1`, commit `3541cde18b2447cd54e7ca6a814eb0d60585c3f9`. All 89 Python files in the wheel are byte-identical to the tag. Every fault below is also in `master` at `627455f` (20 July 2026). Receipts: `CITATIONS.md`.

Method: fixed models or log-probabilities passed straight in, so any coverage gap comes from the conformal step and not from training. Scripts are in this folder, raw output in `output/`.

**Status: not complete.** Parts not tested are listed at the end. Nothing here has been reported to the maintainers by us, and their issue tracker has not been searched.

## Faults

| # | Fault | Where | Measured |
|---|---|---|---|
| 1 | **CQR-R intervals are not conformalised.** The width used to scale the correction is read from the freshly zeroed output tensor, not from the predicted quantiles, so the correction is q_hat x 0.000001. The interval returned is the raw quantile band. | `regression/score/cqrr.py` line 39 (compare line 29) | Raw band covers 57.6%; `CQRR` returns 57.6% at a 90% target. `CQR`, `CQRM`, `CQRFM` on the same data: 90.1% to 90.2%. (`r1_regression.py`) |
| 2 | **ACI is not Gibbs and Candès' method: the level is clipped.** alpha_t is held inside [1/(n+1), 0.9999], so the interval can never become infinite and the long-run guarantee is lost. | `regression/predictor/aci.py` line 197 | gamma 0.05, 3,000 steps, bound on the miss rate 10.6%. Random-walk error: 14.0% (worst run 15.1%); trending series with a fixed model: 11.0%; noise growing 1% a step: 11.0%. Unclipped reference on the same sequences: 9.9% to 10.1%. TorchCP gave no infinite interval in any run; the reference needed them on 14% to 39% of steps. (`r2_aci.py`) |
| 3 | **AgACI is neither ACI nor an aggregation.** (a) Each expert's level is `alpha + gamma (alpha - window error rate)`: nothing is carried from step to step. (b) The weights are computed from the same past intervals for every expert, so all experts get the same weight at every step; the output is their plain average. (c) Infinite bounds are replaced by -99999 and 99999. | `regression/predictor/agaci.py` lines 88, 104 to 113, constructor default `threshold` | Miss rate at a 10% target: 44.5% (trending series), 38.2% (noise growing), 27.5% (random-walk error), 11.1% (one jump in scale), 10.0% (i.i.d.). Identical weights confirmed at every step of every run. (`r2_aci.py`) |
| 4 | **NorABS cannot predict more than one row.** A shape error in the interval formula: any batch of 2 or more rows raises `RuntimeError`. | `regression/score/norabs.py` lines 286 to 289 | Batch of 1 works; 2 and 64 fail, for `variance` and `knn_distance`. (`q2_more.py` part c0) |
| 5 | **NorABS docstring leads to under-coverage.** The argument `data_loader` is documented as "Calibration dataset". Neighbours are then searched in the calibration set itself, so each calibration row is its own nearest neighbour. TorchCP's own example passes the training set instead, which is correct. | `regression/score/norabs.py` lines 224 and 234 | Calibration set as documented: 83.6% (`knn_residual`, k = 5), 84.6% (`knn_label`, k = 5), 88.7% to 88.9% (k = 20), 97.1% (`knn_distance`). Separate set: 90.2% to 90.6%. Same mechanism as crepes fault 1. (`q2_more.py` part c) |
| 6 | **ClusteredPredictor returns every label for classes without a cluster.** The threshold for the "null" cluster is infinity; the authors' code (Ding et al.) uses the ordinary split threshold there, and that line is present but commented out in TorchCP. Valid, but useless where it applies. | `classification/predictor/cluster.py` lines 323 to 324 | 4 classes and 10 classes: every set contains every label (coverage 100%). 100 classes with 30 rare: mean set size 34.3 against 6.0 for plain split, same coverage. 100 balanced classes: fine (6.2). (`q2_more.py` part a) |

Smaller:

| Item | Detail |
|---|---|
| `RC3PPredictor.calculate_threshold` | Fails with `TypeError` when called directly, because `num_classes` is only set inside `calibrate`. |
| `Margin(score_type="identity")` | Overwrites the caller's probability tensor (true-label entries set to -1). `classification/score/margin.py` line 43. |
| Models without parameters | Any predictor built on a parameter-free module raises `StopIteration` (`utils/common.py`, `get_device`). |
| ACI miss indicator | A value exactly on the bound counts as a miss (`>=`, `<=`). |
| `ACIPredictor.train` | Calibrates on the training data itself. |

## What passed

| Part | Result |
|---|---|
| Rank in `calculate_conformal_value` | Exact in 9,835 of 9,835 cases (five levels, n up to 2,000). |
| Classification scores with `SplitPredictor` (LAC, APS, APS not randomised, RAPS, SAPS, Margin, TOPK, EntmaxScore) | 89.7% to 89.9% at 90%; same with one row per call (no batch-size effect). TOPK not randomised 92.0% (ties). |
| Small calibration sets (9 to 99 rows) | On target; infinite threshold when no rank exists. |
| `ClassConditionalPredictor` | 89.6% to 90.9% inside each of four classes. |
| `RC3PPredictor` (through `calibrate` path, `num_classes` set) | 90.3% to 93.2% inside each class. |
| p-values, classification and regression | Uniform: P(p <= 0.1) = 0.100 to 0.103. |
| KNN score | 90.7%. |
| Regression `ABS`, `CQR`, `CQRM`, `CQRFM` | 90.1% to 90.2%. |
| `EnsemblePredictor` (EnbPI), i.i.d. data | 90.1%. |
| Graph: APS, DAPS, SNAPS with `SplitPredictor`; `NAPSPredictor` | 90.1% to 90.7%. |

## Not tested

| Part | Reason |
|---|---|
| `WeightedPredictor` (covariate shift) | Needs an image encoder and domain classifier; code not read. |
| `R2CCP` score | Needs a trained classifier over bins; code read, not run. |
| `ConformalPredictiveDistribution` | Code read only (returns prediction plus signed residuals, unsorted). |
| LLM module (`ConformalLM`) | Needs a language model; `calibrate_configs` read only. |
| Trainers and losses (ConfTr, ConfTS, C-Adapter, SCPO, uncertainty-aware, CF-GNN) | Not read. |
| Metrics (`CovGap`, `WSC`, `SSCV`, p-value criteria and the rest) | Not checked against independent calculations. |
| `ACIPredictor.evaluate` with retraining on | Only the update rule was run, with a fixed model. |
| `EnsemblePredictor` on dependent data | Not run. |
| TorchCP issue tracker | Not searched. |
| Earlier TorchCP notes in `docs/bug_reports_to_send.md` | Not re-checked one by one against 1.2.1 (SAPS coverage is on target here). |
