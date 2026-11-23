# MAPIE 1.6.0: forensic audit

Reissued 23 November 2026. It replaces the version of 22 November, which was called complete when its code read and several modules had not been done.

**Audited:** MAPIE 1.6.0 from PyPI (released 19 October 2026, the latest release), commit `f1cc3d07460b2d7198f557d40d60cd563c51822b`. All four faults below are also present in the `master` branch at commit `e4c4f5f` (19 November 2026).
**Environment:** scikit-learn 1.9.1, SciPy 1.17.1, PyTorch 2.15.0, cvxpy 1.9.3. Tests `base84` to `t11` ran with NumPy 1.26.4; `t13` to `t20` with NumPy 2.5.5, after installing cvxpy upgraded it. The three headline reproductions were re-run under NumPy 2.5.5 and gave the same numbers.
**Evidence:** every finding has a script and its raw output (in `output/`). Dated screenshots, Wayback Machine snapshots and PyPI hashes are indexed in `CITATIONS.md`.

## Summary

MAPIE's core is sound. Split, cross and jackknife conformal regression, CQR and LAC classification met their guarantees in every test, and their thresholds are the exact conformal order statistics. Four faults were reproduced in the current release. Three break a stated guarantee; the fourth makes a default procedure return nothing.

| # | Fault | Measured | Known to MAPIE? |
|---|---|---|---|
| 1 | **ACI clips its working level to [0, 1].** This is not the algorithm of Gibbs and Candès, and their guarantee does not apply to it | Miss rate 13.7%, 19.8% and 33.4% at a 10% target, where the theorem allows 10.6% at most. Unclipped: 10.1% | No report found |
| 2 | **APS and RAPS with `include_last_label=False` under-cover.** The docstring says this option "may result in a coverage higher than 1 - alpha" | APS 81.5%, RAPS 83.6% at a 90% target | Yes. Issue #1021 (7 November 2026); fix proposed in #1024, open |
| 3 | **The default RCPS confidence bound is not the bound of the paper.** To save memory the data are split in two halves, the test statistic is restarted on the second half, and the two maxima are added | Bound below the true risk in up to 19.7% of runs where 10% is allowed, and 11.0% where 5% is allowed. Through the public class: 12.3% where 10% is allowed, 7.1% where 5% is allowed | Yes. Pull request #998 (7 October 2026) explained the fault and fixed it. MAPIE closed it unmerged, saying the code would be refactored, and released 1.6.0 twelve days after it was opened |
| 4 | **The default Holm procedure rejects nothing if any p-value is undefined.** Learn-then-test then returns no threshold | No threshold in 161 of 300 runs; plain Bonferroni on the same data: 0 of 300 | No report found |

"No report found" means none in the issue and pull-request titles available to this audit. It is not proof that nobody reported it.

## Fault 1: ACI's working level is clipped

`mapie/regression/time_series_regression.py`, line 330:

    self.current_alpha[alpha_0] = np.clip(new_alpha_t, 0, 1)

In adaptive conformal inference (Gibbs and Candès, 2021) the update is `alpha + gamma * (target - miss)`, with nothing else. The paper says the level may leave [0, 1]: at or below 0 the interval is infinite. The guarantee (their Proposition 4.1) holds for any sequence at all and bounds the gap between the long-run miss rate and the target by `(max(a, 1 - a) + gamma) / (gamma * T)`. The proof needs each miss to lower the level by the full `gamma * (1 - target)`.

With the clip, a miss when the level is near 0 lowers it by less than that. At a 10% target and gamma 0.05, a miss should cost 0.045 and a cover should earn back 0.005, so nine covers pay for one miss. From a level of 0.01, MAPIE's miss costs 0.01, two covers pay it back, and the next interval can miss again. On a steadily rising series that gives one miss in three.

Test (`t4_timeseries.py`, `t4b_aci_reference.py`, `t9_aci_at_zero.py`): 3,000 steps, gamma 0.05, 10% target, MAPIE's documented loop (`predict`, `adapt_conformal_inference`, `update`). The theorem allows a miss rate between 9.4% and 10.6%.

| Sequence | MAPIE 1.6.0 | Unclipped ACI (pyconformal, and a 12-line loop from the paper) |
|---|---|---|
| i.i.d. noise | 10.0% | 10.0% |
| Noise scale jumps tenfold halfway | 10.1% | 10.1% |
| Errors growing 1% per step | **13.7%** | 10.1% |
| Errors growing 5% per step | **19.8%** | 10.1% |
| Every outcome larger than all before | **33.4%** | 10.1% |

On stationary and mildly dependent data the clip never binds: on three kinds of dependent series ACI's coverage was 88.8% to 90.2% (`t18_ts_dependent.py`). The guarantee that distinguishes ACI, validity under any drift, is the part that is gone.

What MAPIE says about it: the method is named `"aci"`, the class docstring cites Gibbs and Candès, and the example `plot_zaffran2022_comparison.py` says it checks that the implementation "gives the same results" as published ACI work. No MAPIE page states the ACI bound in words. The README says MAPIE implements algorithms that "possess theoretical guarantees".

A self-contained reproduction, using only MAPIE and scikit-learn, is `repro_aci_clip.py`. It prints a miss rate of 13.5% against 10.6% allowed.

Two related traps. `update(X, y, gamma=...)` ignores `gamma`; the level only moves through `adapt_conformal_inference`. And with default settings `predict` raises a `ValueError` once the level is small, so the ACI loop only runs with `allow_infinite_bounds=True`.

## Fault 2: APS and RAPS with `include_last_label=False`

Reported to MAPIE by a user on 7 November 2026 (issue #1021). A fix is proposed in pull request #1024, open on 23 November 2026.

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

## Fault 3: the default RCPS bound

`mapie/risk_control/methods.py`, lines 93 to 115. With `method="rcps"` and no `rcps_bound`, MAPIE uses the Waudby-Smith-Ramdas (WSR) bound. In Bates et al. (2021) the bound comes from one running product over all n calibration losses: the upper bound is the smallest risk value at which the largest value of that product exceeds 1/delta. MAPIE's code has the comment "Split the calculation in two to prevent memory issues". It runs the product separately on each half of the data, takes the largest value on each half, and adds the two (on the log scale). The sum of two separate maxima is larger than the maximum of the whole product, so the threshold is crossed too early and the bound comes out too low. With an odd number of losses the middle one is dropped.

Cause confirmed: a 12-line replica of "two halves, add the maxima" equals MAPIE's output in 28,800 of 28,800 runs (`t10c.py`).

The bound itself (`t10b`, `t10c`, 600 to 2,000 runs per row, MAPIE's own default grid of thresholds). The number is the share of runs in which the upper bound fell below the true risk; it must not exceed delta.

| Losses | True risk | n | delta | MAPIE default | Bound as in the paper |
|---|---|---|---|---|---|
| 0 or 1 | 0.2 | 50 | 0.10 | **19.7%** | 8.5% |
| 0 or 1 | 0.1 | 100 | 0.10 | **18.3%** | 6.3% |
| 0 or 1 | 0.1 | 300 | 0.10 | **14.7%** | 5.5% |
| 0 or 1 | 0.2 | 50 | 0.05 | **11.0%** | 4.3% |
| 0 or 1 | 0.1 | 100 | 0.05 | **10.2%** | 2.8% |
| 0 or 1 | 0.1 | 300 | 0.05 | **8.5%** | 3.5% |
| Continuous on (0, 1) | 0.1 to 0.3 | 50 to 300 | 0.10 | 0.7% to 9.7% | 0.2% to 3.3% |

Of 24 settings with 0/1 losses, 15 were above delta by more than three standard errors. No setting with continuous losses was. The Hoeffding and Bernstein bounds were never below the true risk more often than allowed.

Through the public class, `MultiLabelClassificationController(risk="recall", method="rcps")`, with one label per row so that the loss is 0 or 1 (`t10d_e2e_binary.py`, 1,500 runs each):

| Target | Confidence | Calibration rows | Runs with true recall below target | Allowed |
|---|---|---|---|---|
| recall ≥ 0.9 | 0.90 | 100 | **12.3%** | 10% |
| recall ≥ 0.8 | 0.95 | 100 | **7.1%** | 5% |
| recall ≥ 0.8 | 0.90 | 100 | 11.5% (within noise) | 10% |

With six labels per row (fractional losses) the excess was within noise: 12.8% and 11.7% over 400 runs. `SemanticSegmentationController` inherits the same code.

History. Pull request #998, opened 7 October 2026 by an outside contributor, describes exactly this fault, measures it against the paper authors' reference code, and fixes it with a regression test. A maintainer asked for another pass, then a second maintainer closed it: "We plan to refactor the code of the WSR bound ... we will take your contribution into account but for now we'll close the PR." MAPIE 1.6.0 was released on 19 October 2026 with the fault, and it is in `master` on 19 November 2026. This audit found the fault independently from the code and learned of #998 afterwards. Credit for the first report belongs to that pull request.

`repro_rcps_wsr.py` reproduces it in 12 lines: bound below the true risk in 16.7% of runs, allowed 10%.

## Fault 4: the default Holm procedure and undefined p-values

`mapie/risk_control/fwer_control.py`, line 171. `BinaryClassificationController` and `MultiClassificationLTTController` default to `fwer_method="bonferroni_holm"`. The Holm step picks the smallest remaining p-value with `np.argmin`. If any p-value is NaN, `np.argmin` returns the NaN, the comparison with the level is false, and the loop stops. Nothing is rejected, so no threshold is valid.

A NaN p-value arises whenever a threshold has no observations for the risk, for example precision at a threshold above every predicted probability. That is common at the top of MAPIE's default grid (0 to 0.99).

The smallest example (`t11c_holm_nan.py`): p-values `[1e-9, 1e-9, 1e-9, NaN]`, delta 0.1. Bonferroni rejects three. MAPIE's Holm rejects none. Holm can never be less powerful than Bonferroni, so this is a fault, not a design choice.

In use (`t11b_ltt.py`, `t11c_holm_nan.py`, `t19_multiclass_ltt.py`):

| Task | n | Default (Holm): no threshold | Bonferroni: no threshold |
|---|---|---|---|
| Binary, precision ≥ 0.6 | 2,000 | 161 of 300 | 0 of 300 |
| Binary, precision ≥ 0.6 | 500 | 251 of 300 | 7 of 300 |
| Three classes, precision ≥ 0.6 each | 1,000 | 188 of 200 | 4 of 200 |

In the first row the 161 runs with no threshold are exactly the 161 runs that contained a NaN p-value. Validity is not affected: every threshold that was returned met its target. The cost is that the default returns "no valid threshold" for a target the model can reach. This explains the earlier unexplained result (no threshold in 348 of 400 runs).

`fwer_method="fixed_sequence"` returned no threshold for precision in 300 of 300 runs at every sample size: it starts at the highest threshold, which has the fewest observations.

## Smaller findings

| Finding | Evidence |
|---|---|
| With float32 `X` and float32 `y` centred near zero, `SplitConformalRegressor` raises "get_conformity_scores and get_estimation_distribution ... are not consistent" on 200 of 200 random datasets. With `y` centred at 50 or 1,000, or with either input in float64, it runs. Standardised float32 targets are ordinary in practice. The same error was reported in #321 (2023) and #557 (December 2024); both were closed with a workaround | `t17c_float32.py` |
| `allow_infinite_bounds=True` is ignored with the default symmetric score: MAPIE raises instead of returning an infinite interval | `t8b.py`, `t2_exact.py` (290 of 290 cases) |
| At exactly `n = 1/alpha - 1` calibration points (9 points for 90%) a valid finite interval exists, the largest score, but MAPIE refuses | `t2_exact.py` |
| `spiegelhalter_p_value` is one-sided. It rejected in 96.6% of runs when the model under-predicts and in 0% when it over-predicts by the same amount. The docstring describes it only as the probability of the statistic under perfect calibration | `t14_metrics.py` |
| `ConditionalExpectedRiskController` with its default learning rate (1e-4) barely moves from its starting point: it kept 96% of labels and reached a risk of 0.002 against a target of 0.10. With a learning rate of 1e-2 it reached 0.097. Safe, but the default is of no use. Prediction retrains a model for every test row: about 0.5 s per row | `t20_aacrc.py` |
| `expected_calibration_error` with 50 bins differs from a hand calculation in the third decimal (0.1092 against 0.1081); with 10 bins it agrees. Not investigated further | `t14_metrics.py` |
| A target `y` of shape (n, 1) makes `conformalize` raise a broadcasting error | `t17_inputs.py` |
| Data-splitting helper gives 800 / 99 / 101 rows for 0.8 / 0.1 / 0.1 of 1,000 | `t7_misc.py`; issue #1018 |
| Venn-Abers emits 658 "invalid value encountered in divide" warnings in one fit and predict. Output is correct | `t6_risk_va.py`; issue #1030 |
| `TopLabelCalibrator.predict_proba` returns NaN for every class except the top one, so rows do not sum to 1. This follows from the method, but it breaks code that expects probabilities | `t15_calibration.py` |

## What passed

| Area | Test | Result |
|---|---|---|
| 84 coverage configurations on i.i.d. data (`base84.py`) | Split regression with three scores, CQR, cross-conformal, jackknife+-after-bootstrap, split and cross classification | 78 met their target; 6 refused to run because the calibration set was too small, which is safe |
| Threshold exactness (`t2_exact.py`) | Split regression and LAC against the exact rank | 907 of 907 and 431 of 431 exact |
| The two shared quantile functions, called as MAPIE calls them (`t8_code_read.py`, `t8b.py`) | 3,000 calibration sizes by 13 levels | 37,581 of 37,581 exact, both functions |
| Width minimisation (`t8_code_read.py`) | Skewed errors, absolute and gamma scores, 30 to 300 calibration points | Coverage held in every case |
| CQR with 5 to 100 calibration points (`t3_cqr_small.py`) | Symmetric and asymmetric | No under-coverage |
| Conformal risk control (`t6_risk_va.py`) | Multi-label recall, target 0.9 | Mean recall 0.908 |
| RCPS with Hoeffding and Bernstein bounds (`t10*`) | Bound and end to end | Within delta everywhere |
| Learn-then-test validity (`t11b_ltt.py`, `t19_multiclass_ltt.py`) | Precision and recall, binary and three-class, three FWER methods | True metric below target in at most 7% of runs, 10% allowed |
| Conditional conformal regressor (`t13_conditional.py`) | Two groups with fivefold different noise, 500 runs | Group coverage 90.4% and 90.6% at 90%; the ordinary split regressor gives 100% and 67% |
| Conditional conformal classifier | LAC and APS, two groups | LAC 90.2% and 90.6%; APS 98.1% and 97.1% (over-covers) |
| Exchangeability tests on i.i.d. data (`t12_exchangeability.py`, 600 runs, 5% level) | Jumper and plug-in martingales, permutation p-value, three sequential Monte Carlo tests; continuous and heavily tied scores | Rejection rates 0.0% to 4.8% (i.i.d.: all six tests; tied scores: the first three tests; the run stopped before the three sequential tests, which rejected 0 or 1 of 20 trial runs). All six rejected a drifting series in every one of 40 trial runs |
| Calibrators (`t15_calibration.py`) | Top-label (sigmoid, isotonic) and Venn-Abers (prefit, inductive, cross), two and four classes | Top-label calibration error fell from 0.078 to about 0.02 and from 0.134 to about 0.025. Venn-Abers output identical to the venn-abers package |
| Metrics (`t14_metrics.py`) | Coverage, width, Winkler score, size-stratified coverage, set size, ECE with 10 bins, Spiegelhalter statistic against hand calculations; Kolmogorov-Smirnov and Kuiper tests against their 5% level | All agree; tests reject 3.6% to 5.0% of calibrated runs |
| EnbPI on dependent data (`t18_ts_dependent.py`) | Three kinds of series; with and without update, mean and median, `optimize_beta` | Stationary and volatility-clustered series: 87% to 93%. Under a doubling of noise: 81% with update, 62% without. EnbPI promises only approximate coverage under conditions, so this is the method, not a fault |
| Input handling (`t17_inputs.py`) | pandas (including shuffled indexes), sparse, lists, one test row, three levels at once and unsorted, string and non-contiguous labels, a class missing from calibration, weights, groups | Same results as NumPy in every case; NaN inputs are rejected with a clear error |
| Old open issues retested (`t7_misc.py`) | #710, #542, #484 | All three work in 1.6.0 |

## Coverage of this audit

| Part of MAPIE | Status |
|---|---|
| `utils.py` quantile functions; `conformity_scores/regression.py` | Code read and tested |
| `regression/regression.py` (split, cross, jackknife+-after-bootstrap) | Tested. Code not read line by line |
| `regression/quantile_regression.py` (CQR, cross-CQR) | Tested. Code not read line by line |
| `regression/time_series_regression.py` (EnbPI, ACI) | Code read and tested |
| `classification.py`, `conformity_scores/sets` (LAC, APS, RAPS, top-k) | APS code read; all tested. RAPS and top-k code not read line by line |
| `risk_control/methods.py`, `fwer_control.py`, `_base_controller.py` | Code read and tested |
| `risk_control` controllers (binary, multi-class, multi-label) | Tested |
| `risk_control/semantic_segmentation.py` | Not run. It inherits the multi-label code, including fault 3 |
| `risk_control/adaptive_conformal_risk_control.py` | Code read; one test with a constant feature map. Its conditional guarantee was not tested |
| `conditional_conformal_prediction.py` | Tested with group indicators. Kernel feature maps (`infinite_params`) not tested |
| `exchangeability_testing` | Martingale code read; all tests run. `RiskMonitoring` not tested |
| `calibration.py`, `_venn_abers.py` | Tested |
| `metrics` | Tested, except `hsic`, `coverage_width_based` and the three functions in `metrics/conditional.py` |
| `experimental` module, Mondrian use, `subsample.py` | Not tested |
| Issue tracker | Titles for 218 of the issues, from an archive that proved out of date (see corrections). Full text read for #557, #998 only |

## Corrections to the version of 22 November

- **Issue statistics are withdrawn.** They came from an archive of the tracker that is out of date: it lists #557 and #998 as open, and both are closed. The live tracker showed 24 open issues and 16 open pull requests on 23 November 2026, not 53 open issues. The figures "7 bug-labelled issues open, all older than two years" and "53 open, 28 older than two years" must not be used until recounted from the live tracker. Dates on which individual issues were opened are not affected.
- **Fault 1, mechanism.** The earlier text said a level stuck at zero forgets its debt. At level 0 MAPIE does return an infinite interval. The fault is that a miss near 0 is under-charged, as described above. The measured miss rates are unchanged.
- **Learn-then-test.** The "no threshold in 348 of 400 runs" result is explained by fault 4.

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

## How to reproduce

    pip install mapie==1.6.0 pyconformal venn-abers scikit-learn torch cvxpy
    python repro_aci_clip.py            # fault 1, 1 minute
    python repro_rcps_wsr.py            # fault 3, 1 minute
    python t11c_holm_nan.py             # fault 4, 1 minute
    python t1_classification.py 200     # fault 2
    python base84.py 200                # about 30 minutes
    python t10c.py 600 400              # about 25 minutes
    python t12_exchangeability.py 600 400   # about an hour

The remaining scripts take one argument or none; each starts with a comment saying what it tests.
