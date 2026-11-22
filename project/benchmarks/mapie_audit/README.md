# MAPIE 1.6.0: forensic audit

22 November 2026. Audited: MAPIE 1.6.0 from PyPI (released 19 October 2026, the latest release), with scikit-learn 1.9.1 and NumPy 1.26.4, and the repository at commit e4c4f5f (19 November 2026). Every finding below has a script in `scripts/` and its raw output in `output/`.

## Summary

MAPIE's core is sound: split, cross and jackknife conformal regression, CQR and LAC classification met their guarantees in every test, and their thresholds are the exact conformal order statistics. Two faults break a stated guarantee in the current release, both reproduced:

| # | Fault | Measured | Status |
|---|---|---|---|
| 1 | **ACI clips its working level to [0, 1].** The method's long-run guarantee is proved for the unclipped update; with clipping it fails on hard sequences | Miss rate 13.7%, 19.8% and 33.4% at a 10% target on three sequences where the theorem allows at most 10.6%. An unclipped ACI gave 10.07% on all three | In the release and on master. Not reported, as far as I can find |
| 2 | **APS and RAPS sets with `include_last_label=False` under-cover.** The documentation says this option "may result in a coverage higher than 1 - alpha" | APS: 81% at a 90% target, 69% at 80%. RAPS: 84% and 70%. Stable from 50 to 1,000 calibration points | Reported as issue #1021 in November 2026; fix proposed in PR #1024; not released |

Everything else is waste, noise or history: sets larger than needed by default, warnings, slow paths, and a record of correctness fixes that is long but mostly prompt.

## What passed

| Area | Test | Result |
|---|---|---|
| 84 coverage configurations on i.i.d. data (`base84.py`, 200 repetitions) | Split regression with three scores, CQR, cross-conformal, jackknife+-after-bootstrap, split and cross classification | 78 met their target; 6 refused to run because the calibration set was too small, which is safe. 0 failures |
| Threshold exactness (`t2_exact.py`) | Split regression against the exact rank, 67 calibration sizes by 9 levels, with and without `allow_infinite_bounds` | 907 of 907 finite thresholds exact. 0 too low, 0 too high |
| | LAC classification, 65 sizes by 9 levels | 431 of 431 exact |
| CQR with 5 to 100 calibration points (`t3_cqr_small.py`) | Symmetric and asymmetric, 90% and 80% | No under-coverage. Too-small calibration sets are refused |
| EnbPI, documented predict/update loop, i.i.d. data (`t4_timeseries.py`) | 5 runs of 400 steps, 90% target | 88.7% to 91.0% |
| Conformal risk control, multi-label recall (`t6_risk_va.py`) | 300 runs, target 0.9 | Mean recall 0.908 |
| Venn-Abers | Against the venn-abers package | Identical output |
| Old open issues retested (`t7_misc.py`) | #710 (fit twice), #542 (leave-one-group-out), #484 (several levels in time series) | All three work in 1.6.0; the issues are still open |

The under-coverage reported for asymmetric CQR on small calibration sets (PR #978: 87.5% at 90% with 15 points) does not occur in 1.6.0: the release refuses those sizes.

## Fault 1: ACI's working level is clipped

`mapie/regression/time_series_regression.py`, line 330:

    self.current_alpha[alpha_0] = np.clip(new_alpha_t, 0, 1)

In adaptive conformal inference (Gibbs and Candès, 2021) the working level moves down after every miss and may go below zero, where the interval is infinite. The theorem, which holds for any sequence at all, bounds the gap between the long-run miss rate and the target by `(max(a, 1 - a) + gamma) / (gamma * T)`. Its proof needs the level to be free: a level stuck at zero "forgets" how many misses it still owes.

Test (`t4_timeseries.py`, `t4b_aci_reference.py`): 3,000 steps, gamma 0.05, 10% target, MAPIE's documented loop (`predict`, `adapt_conformal_inference`, `update`). The theorem allows a miss rate between 9.4% and 10.6%.

| Sequence | MAPIE 1.6.0 | Unclipped ACI (pyconformal, and a 12-line loop from the paper) |
|---|---|---|
| i.i.d. noise | 10.0% | 10.0% |
| Noise scale jumps tenfold halfway | 10.1% | 10.1% |
| Errors growing 1% per step | **13.7%** | 10.1% |
| Errors growing 5% per step | **19.8%** | 10.1% |
| Every outcome larger than all before | **33.4%** | 10.1% |

On stationary data nothing is lost, which is why ordinary tests do not catch it. The guarantee that distinguishes ACI, validity under any drift, is the part that is gone. The fix is to remove the clip and treat a level at or below zero as an infinite interval, which the prediction path already supports.

A related trap: `update(X, y, gamma=...)` accepts `gamma` and ignores it, with only a warning. The level adapts only through `adapt_conformal_inference`.

## Fault 2: APS and RAPS with `include_last_label=False`

Test (`t1_classification.py`): 5 classes, labels drawn from the model's own probabilities (so the model is perfect and the data exchangeable), 200 repetitions, 1,000 test points.

| Score | `include_last_label` | Coverage at 90% | Coverage at 80% | Mean set size at 90% |
|---|---|---|---|---|
| APS | `True` (default) | 95.1% | 88.4% | 3.38 |
| APS | `False` | **81.5%** | **68.8%** | 2.39 |
| APS | `"randomized"` | 90.3% | 80.4% | 2.95 |
| RAPS | `True` (default) | 95.7% | 89.0% | 3.53 |
| RAPS | `False` | **83.6%** | **70.4%** | 2.53 |
| RAPS | `"randomized"` | 90.3% | 80.3% | 2.97 |
| LAC | | 90.3% | 80.3% | 2.82 |
| Top-k | | 93.8% | 89.2% | 3.50 |

(200 calibration points; 50 and 1,000 give the same picture.)

The cause is a mismatch: calibration scores are computed one way and the prediction rule drops the label that crosses the threshold. The docstring of `APSConformityScore.get_prediction_sets` says that `True` or `False` "may result in a coverage higher than `1 - alpha`". For `False` the opposite is true, by 9 to 11 points.

The same table shows the cost of the defaults: APS and RAPS cover 95% to 96% when asked for 90%, with sets about 15% to 20% larger than the randomized rule that hits the target.

## Smaller findings

| Finding | Evidence |
|---|---|
| Data-splitting helper gives 800 / 99 / 101 rows for 0.8 / 0.1 / 0.1 of 1,000 | `t7_misc.py`; issue #1018, open |
| Venn-Abers emits 658 "invalid value encountered in divide" warnings in one fit and predict | `t6_risk_va.py`; issue #1030, open. Output is correct |
| At exactly `n = 1/alpha - 1` calibration points (9 points for 90%) a valid finite interval exists, the largest score, but MAPIE refuses | `t2_exact.py`: 9 of 9 such cases. Conservative, harmless |
| `allow_infinite_bounds=True` did not return an infinite interval in any of 290 cases where the rank exceeds the calibration size; it raises instead | `t2_exact.py` |
| Learn-then-test precision control returned no threshold in 348 of 400 runs for a target the model can reach; the 52 thresholds it did return were all valid | `t6_risk_va.py`. Not investigated further; may be my configuration. Low confidence |

## Speed and memory (measured 20 to 21 November 2026, same inputs on both sides)

| Method | MAPIE 1.6.0 | pyconformal | Factor |
|---|---|---|---|
| Venn-Abers, 1,000,000 calibration points | 61.9 s | 0.33 s | 189x, identical output |
| EnbPI backtest, per step | 5.5 ms | 0.012 ms | 400x to 470x |
| EnbPI one row at a time, per step | 5.8 ms | 2.1 to 2.3 ms | about 2.5x |
| ACI backtest, per step | 0.41 ms | 0.0002 ms | about 2,000x |
| ACI one row at a time, per step | 0.57 ms | 0.087 ms | 6x to 8x |
| CV+ prediction, 20,000 test rows | 1.95 s and 4.99 s | 0.115 s and 0.184 s | 17x and 27x |
| CV+ prediction, 20,000 training by 20,000 test rows | killed for memory on a 7 GB machine | 0.28 s | |
| CQR, 1,000,000 test rows | 19.7 ms | 3.2 ms | 6x, identical intervals |

Memory use of the cross-conformal methods has been an open issue (#160) since May 2022.

## The record: correctness fixes in MAPIE's changelog

| Release | Date | Fix |
|---|---|---|
| 0.6.0, 0.6.1 | Jan 2023 | Classification with very low scores |
| 0.6.3, 0.6.5 | Mar, Jun 2023 | Labels not starting at 0; calibration set with fewer labels |
| 0.8.4 | Jun 2024 | Quantile formula, to ensure valid coverage for any number of calibration points (#452, reported May 2024) |
| 0.8.6 | Jun 2024 | Quantile formula again: infinite intervals and asymmetric scores |
| 0.9.2 | Jan 2025 | EnbPI intervals centred on the wrong prediction (#495, reported Jul 2024, fixed by an outside contributor) |
| 1.2.0 | Nov 2025 | Block bootstrap left rows out of the test set (#790) |
| 1.3.0 | Feb 2026 | Risk control: computed threshold not optimal. ACI `update` |
| 1.4.1 | Jun 2026 | Width minimisation (`optimize_beta`) did nothing (#484 Jul 2024, #588 Jan 2025). Binary risk controller tie-breaking (#942) |
| 1.6.0 | Oct 2026 | Classification threshold one rank too high (#479, opened Jul 2024). CQR threshold one rank too high (#991). CQR counted twice the calibration points, letting too-small sets through. Cross-CQR computed the wrong quantity. RCPS bound used the wrong variance. Martingale test p-value (#984). Finite bound returned where none is valid (#974, #980) |

How fast are reports handled? From an archive of 218 of MAPIE's issues (not the complete tracker):

- 48 carry the bug label. 41 are closed, with a median of 9 days to close; 29 closed within 30 days; 3 took more than 180 days.
- 7 bug-labelled issues are open, all older than two years (806 to 1,173 days). Most concern documentation or tooling; one concerns coverage with infinite intervals (#462, June 2024).
- The slow ones that mattered: #479, 725 days; #588, 495 days.
- 53 issues are open in all, 28 of them older than two years; the oldest dates from July 2021.

## Not tested

Conditional conformal prediction, exchangeability tests, semantic segmentation and multi-class risk control, Mondrian, EnbPI on dependent data, the experimental module.

## How to reproduce

    pip install mapie==1.6.0 pyconformal venn-abers scikit-learn
    cd scripts
    python base84.py 200          # about 30 minutes
    python t1_classification.py 200
    python t2_exact.py
    python t3_cqr_small.py 300
    python t4_timeseries.py
    python t4b_aci_reference.py
    python t6_risk_va.py
    python t7_misc.py
