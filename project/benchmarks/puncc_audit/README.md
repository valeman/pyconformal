# PUNCC 0.9.3: forensic audit

23 November 2026.

**Audited:** puncc 0.9.3 from PyPI (released 23 June 2026, the latest release), tag `v0.9.3`, commit `7a0db784b9f3418bb1068ae4de2ba1902ee2b194`. The four files cited below are byte-identical in the PyPI wheel, at that commit and in `main` (commit `29a1219`, 20 August 2026).
**Environment:** Python 3.13, NumPy 1.26.4, scikit-learn 1.9.1.
**Evidence:** every finding has a script here and its raw output in `output/`. Receipts are in `../CITATIONS_crepes_puncc.md`.

## Summary

PUNCC's regression side is the cleanest of the three libraries audited: every method met its level and the threshold is the exact conformal rank at all 9,835 sizes tested. Two faults on the classification and object-detection side are severe. Both produce sets that are far below their stated coverage with no error or warning, both are in the current release and in `main`, and both have been in the code for about two years.

| # | Fault | Measured | Known to PUNCC? |
|---|---|---|---|
| 1 | **APS and RAPS cap the size of a prediction set at the number of test rows in the call.** One row per call gives sets of one label | Coverage 45% at a 90% target with one row per call, 68% with two, 89% with five | No report found |
| 2 | **Box-wise object detection uses the wrong sign for the two "max" sides of the box.** If the detector's boxes are too small, the "outer" box is made smaller still | True box inside the outer box in 0% of cases at a 90% target, additive and multiplicative. With symmetric errors the additive method looks correct (91%) and the multiplicative one gives 86% | No report found |
| 3 | **EnbPI and AdaptiveEnbPI crash when residuals are updated after every observation** (`s=1`) | `TypeError` and `IndexError`; `s=2` and above work | No report found |

"No report found" means none among the 40 issue titles on the tracker on 23 November 2026. Two open issues concern nearby code: #101 (RAPS randomisation over-covers when `lambd > 0`) and #100 (CQR fails for a single test point).

## Fault 1: APS and RAPS depend on the batch size

`deel/puncc/api/prediction_sets.py`, lines 163 and 192:

    pred_len = shape[0]            # number of test rows in this call
    ...
    L = b.minimum(b.sum(below_tau, axis=-1) + 1, pred_len)

`L` is the number of classes to put in each set. It is capped at `pred_len`, the number of rows passed to `predict`, where the number of classes is meant. A call with one row can only return a one-label set, a call with two rows at most two labels, and so on. With at least as many rows as classes the cap does not bind.

`p0_raps_rows.py`: ten classes, a perfectly calibrated model, 1,000 calibration rows, 90% target, the same fitted object asked for sets in batches of different sizes.

| Rows per call | APS | APS, `rand=False` | RAPS | LAC |
|---|---|---|---|---|
| 1 | **45.4%** | **45.5%** | **45.5%** | 89.2% |
| 2 | **68.1%** | **68.3%** | **68.3%** | 89.2% |
| 5 | 88.8% | 93.8% | 89.5% | 89.2% |
| 20 | 89.8% | 96.1% | 90.4% | 89.8% |
| 1,000 | 89.8% | 96.1% | 90.3% | 89.8% |

One row per call is how a deployed model is normally served. A user who validates on a test set in one call sees 90% and then gets 45% in production.

The same line is in v0.7.4 (December 2023), so the fault is about two years old. `repro_aps_one_row.py` reproduces it in 20 lines: 89.5% in one call, 45.1% one row at a time.

## Fault 2: box-wise object detection

Boxes are `(x_min, y_min, x_max, y_max)`. For the outer box to contain the true box, each side must be pushed outwards by a margin that covers how far the truth lies outside the prediction on that side:

- `x_min`, `y_min`: the margin must cover `prediction - truth`.
- `x_max`, `y_max`: the margin must cover `truth - prediction`.

PUNCC computes `prediction - truth` for all four sides (`nonconformity_scores.py`, lines 229 and 403) and then builds the outer box as `x_min - q0, y_min - q1, x_max + q2, y_max + q3` (`prediction_sets.py`, lines 426 to 427 and 473 to 476). For the two max sides the sign is reversed. When the predicted `x_max` falls short of the truth, the scores are negative, the margin `q2` is negative, and the "outer" box is pulled further inside.

With errors symmetric around zero, `prediction - truth` and `truth - prediction` have the same distribution, so the fault does not show. Real detectors are not symmetric.

`p3_boxes.py`: 300 calibration boxes, 1,000 test boxes, 60 repetitions, 90% target. The last column is the method computed from its definition on the same data.

| Detector's errors | PUNCC additive | PUNCC multiplicative | By definition (additive) |
|---|---|---|---|
| Symmetric | 90.9% | **85.8%** | 91.2% |
| Boxes too small | **0.0%** | **0.0%** | 90.5% |
| Boxes too large | 95.7% | 95.7% | 91.3% |
| Boxes shifted right and down | 95.2% | 95.2% | 90.9% |

Per side, in the "too small" case: `x_min` and `y_min` are covered in 97.6% of cases, `x_max` and `y_max` in 0.0%. PUNCC's own metric `object_detection_mean_coverage` reports the same 0.0%. The multiplicative method under-covers even with symmetric errors, because the scale it divides by depends on the same errors.

PUNCC's theory page states the same thing as the code: scores `prediction - truth` for all four coordinates (`docs/source/theory_overview.rst`, line 532). The code has been this way since object detection was added on 12 January 2024 (commit `6e0a8f8`).

`repro_boxes.py` reproduces it in 20 lines.

## Fault 3: EnbPI with `s=1`

`EnbPI.predict(X, alpha, y_true=y, s=1)` raises `TypeError: 'numpy.float64' object is not iterable`; `AdaptiveEnbPI` raises `IndexError`. Updating after every observation is the natural setting for one-step-ahead forecasting. With `s` of 2, 10 or 50, and with no feedback, both methods run and cover 89.6% to 89.9% on i.i.d. data (`p2_rest.py`, part (a)).

## Smaller findings

| Finding | Evidence |
|---|---|
| One calibration point raises `TypeError: object of type 'numpy.float64' has no len()` where a clear message is given for 2 to 8 points | `p1_regression.py`, part (d) |
| `SplitCAD` refuses with "All coordinates of [q] must be in the open interval (0, 1)" at exactly `n = 1/alpha - 1` calibration points, where the largest score is a valid threshold | `p4b_cad_exact.py` |
| APS with `rand=False` covers 96.1% at a 90% target. That follows from the method, which keeps the boundary label | `p0_raps_rows.py` |

## What passed

| Area | Test | Result |
|---|---|---|
| Split conformal regression (`p1_regression.py`) | 9 to 1,000 calibration points, alpha 0.1 and 0.05 | 89.8% to 90.1% and 95.0% to 95.8%; refuses when the set is too small |
| Exact rank | Calibration sizes 1 to 2,000 at five levels | Exact in 9,835 of 9,835 cases |
| Locally adaptive CP, CQR | Fixed predictions and with models | 88.9% to 90.1% at 90% |
| CV+ | K = 5 and K = 20, linear model and forest | 90.2% to 91.2% (guarantee: at least 80%) |
| Leverage-weighted CP | Linear model | 91.1% |
| Weights | Equal, decaying, random | 90.2% to 91.1%; equal weights give the unweighted interval |
| EnbPI and AdaptiveEnbPI on i.i.d. data | No feedback, `s` = 2, 10, 50 | 89.6% to 89.9% |
| LAC | Any batch size | 89.2% to 89.8% |
| Class-wise LAC (`p2_rest.py`, part (b)) | Four classes of very different difficulty | 89.9% to 90.4% inside each class, where plain LAC gives 46% to 100% |
| Anomaly detection (`p4_cad.py`, `p4b_cad_exact.py`) | Fixed score, 19 to 500 calibration points, 2,000 runs | Share of normal points flagged equals the exact conformal rate to three decimals |
| Metrics | Coverage, sharpness, average coverage error, set coverage and size against hand calculations | All agree |

## Coverage of this audit

| Part | Status |
|---|---|
| `api/utils.py` quantiles, `api/calibration.py`, `api/nonconformity_scores.py`, `api/prediction_sets.py` | Code read and tested |
| `regression.py`: SplitCP, LocallyAdaptiveCP, CQR, CVPlus, LeverageWeightedCP, EnbPI, AdaptiveEnbPI | Tested. EnbPI's prediction loop read |
| `classification.py`: LAC, ClasswiseLAC, APS, RAPS | Code read and tested |
| `anomaly_detection.py`, `object_detection.py` | Code read and tested |
| `metrics.py` | Tested, except the classwise and object-detection area functions and `iou` |
| EnbPI on dependent data | Not tested |
| `api/experimental.py` (PyTorch predictor), TensorFlow, PyTorch and JAX backends, `plotting.py`, Hungarian matching, `CrossValCpAggregator` used directly, Bonferroni helpers | Not tested |
| Issue #101 (RAPS randomisation) | Read, not re-run |

## How to reproduce

    pip install puncc==0.9.3 scikit-learn
    python repro_aps_one_row.py     # fault 1
    python repro_boxes.py           # fault 2
    python p0_raps_rows.py          # fault 1 in full
    python p3_boxes.py              # fault 2 in full
    python p1_regression.py 300     # regression, exact rank
    python p2_rest.py 300           # fault 3 and the rest
