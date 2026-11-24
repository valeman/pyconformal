# Adaptive conformal predictive distributions: literature check and verdict

24 November 2026. Method: 40 arXiv listing queries (279 records, 151 passed a keyword screen, titles read; 12 abstracts read; 6 papers opened), plus the web searches of 23 November. Not searched: journals without arXiv copies, Google Scholar citation trails, conference workshops. Papers read in part, not line by line.

## Verdict

**The core of idea 1 is already published.** MultiQT (Ding, Gibbs and Tibshirani, arXiv:2512.23671, December 2025) calibrates quantile forecasts at many levels at once for arbitrary sequences, keeps the quantiles ordered, and proves a no-regret bound. Two follow-ups extend it. Our prototype is a weaker version of the same idea, and its "sorted" variant is the construction MultiQT's Proposition 2 shows can fail.

**In head-to-head tests MultiQT is at least as good as our tracker, and better under strong drift and on both panels** (tables below).

**What may still be ours, each modest:**

| Possible contribution | Status |
|---|---|
| Running the multi-level tracker on the PIT scale: no step size in data units, no bounded-error assumption, bound (1 + gamma)/(gamma n) for any sequence | Follows from MultiQT applied to PIT values (errors bounded by 1). A corollary, not a new method. Empirically it loses to outcome-scale MultiQT under location drift. |
| Predictive distributions for panels: cross-sectional weighted base (W-TQA weights) recalibrated per target with feedback on a fraction of rounds | Not found in the literature. But the recalibrator that works best in it is MultiQT. A combination and an application, with strong numbers. |
| Benchmark of online recalibrators for predictive distributions on panels | Not found. |

## Closest prior work

| Paper | What it does | Relation |
|---|---|---|
| Ding, Gibbs, Tibshirani, "Calibrated multi-level quantile forecasting", arXiv:2512.23671 (2025) | MultiQT: offsets added to base quantiles, lazy gradient step, isotonic projection; calibrated at every level for adversarial sequences with bounded errors; ordered; no regret; handles delayed feedback. Code: github.com/tiffanyding/multiQT | Direct prior art for idea 1 |
| Ochoa Rivera, Tewari, "Online conformal prediction: enforcing monotonicity via online optimization", arXiv:2605.12668 (2026) | Nested online prediction sets across coverage levels | Same problem |
| Liu, Dobriban, Orabona, "A black-box reduction from regret to multi-level coverage", arXiv:2611.12059 (November 2026) | Isotonic projections with any online learner; better dependence on the number of levels | Same problem, newest |
| Angelopoulos, Candès, Tibshirani (2023), quantile tracker; Gibbs, Candès (2021), ACI | One level | Our "one tracker per level" is ACI run at each level |
| Angelopoulos, Jordan, Tibshirani, "Gradient equilibrium in online learning", arXiv:2501.08330 | Theory behind the trackers | Background |
| Deshpande, Marx, Kuleshov, arXiv:2302.12196; Marx, Kuleshov, Ermon, arXiv:2409.19157; Hu, Tian, Yang, arXiv:2607.19689 | Online recalibration of predictive distributions against an adversary | Different mechanism, same goal |
| Vovk, "Enhancement of prediction algorithms by betting", arXiv:2105.08669; "Protected probabilistic regression" | Betting on the PIT to protect a predictive distribution after a change | Different mechanism, same goal |
| Jonkers et al., arXiv:2404.15018; Jonkers, Ziegel, arXiv:2606.11044 | Conformal predictive systems under covariate and other shifts through weights | Needs the shift to be modelled; not online adaptation |
| Allen, Pescara, Ziegel, arXiv:2510.26914; Allen et al., arXiv:2503.03841 | Predictive systems from residual distributions and from isotonic distributional regression | Exchangeable setting |
| Tu, Giesecke, arXiv:2605.17705; Lin et al. (2022); Batra et al., arXiv:2310.02863 | Panels: intervals only | Panel side of idea 2 |

Nothing found that produces predictive distributions for panels with late or missing feedback.

## Head to head, one series (`compare_single.py`, 3 seeds, 19 levels, 3,000 steps)

Worst-level calibration error, then pinball loss.

| Stream | PIT tracker, lazy isotonic | MultiQT as published | Online recalibration | Betting |
|---|---|---|---|---|
| i.i.d. | 0.001, 0.3015 | 0.002, 0.3078 | 0.009, 0.2989 | 0.008, 0.2996 |
| Noise x5 | 0.001, 0.9029 | 0.003, 0.9187 | 0.009, 0.8977 | 0.016, 0.9004 |
| Noise grows | 0.002, 2.3415 | 0.003, 2.3778 | 0.010, 2.3132 | 0.029, 2.3246 |
| Bias drifts | 0.002, 0.3019 | 0.003, 0.3079 | 0.010, 0.2989 | 0.024, 0.3004 |
| Random-walk error | 0.005, 0.1333 | 0.009, 0.0480 | 0.103, 0.2271 | 0.160, 0.2020 |
| Regime switches | 0.002, 0.7756 | 0.016, 0.7783 | 0.022, 0.8073 | 0.021, 0.8033 |
| SGSC mean load (real) | 0.001, 0.0184 | 0.000, 0.0101 | 0.039, 0.0224 | 0.042, 0.0201 |
| M5 store total (real) | 0.003, 102.07 | 0.004, 102.26 | 0.015, 101.32 | 0.013, 101.38 |

PIT-scale tracking is about 2% sharper than MultiQT when the scale changes, and far worse when the location drifts (random walk, SGSC), because it can only read the base distribution at another level, while MultiQT shifts the quantiles themselves. MultiQT is also steadier over short stretches (worst 250-step error 0.03 to 0.10 against 0.05 to 0.20) and never returns an infinite quantile. The lazy isotonic rule and the first "sorted" prototype give the same numbers to three decimals except on random-walk errors, where sorting is worse (0.012 against 0.005).

## Head to head, panels (`compare_panel.py`, 10 splits, similarity weights, feedback on 20% of rounds)

| Panel | Recalibrator | Per-unit calibration error | Pinball | Central 90% coverage, worst 10% of units | Infinite quantiles |
|---|---|---|---|---|---|
| M5 | none | 0.117 | 0.2073 | 0.731 | 0.8% |
| M5 | PIT tracker, lazy isotonic | 0.065 | 0.1915 | 0.886 | 0.7% |
| M5 | Online recalibration (empirical PIT) | 0.052 | 0.1897 | 0.784 | 0.5% |
| M5 | Betting enhancement | 0.092 | 0.1948 | 0.780 | 0.9% |
| M5 | MultiQT | 0.030 | 0.1804 | 0.875 | 0 |
| SGSC | none | 0.103 | 0.0410 | 0.719 | 1.4% |
| SGSC | PIT tracker, lazy isotonic | 0.058 | 0.0384 | 0.862 | 1.1% |
| SGSC | Online recalibration (empirical PIT) | 0.045 | 0.0377 | 0.782 | 0.2% |
| SGSC | Betting enhancement | 0.081 | 0.0401 | 0.747 | 1.5% |
| SGSC | MultiQT | 0.026 | 0.0392 | 0.859 | 0 |

MultiQT halves our calibration error on both panels and wins pinball on M5. On SGSC online recalibration has the best pinball loss but poor tail coverage.

Caveats on the comparators: MultiQT is our re-implementation from the paper (their code not run), with their step-size rule including its floor of 0.1 in outcome units; online recalibration uses the running frequency as its subroutine, a simplification of the paper's algorithm; the betting method is the Simple Jumper with the paper's settings.

## Tests still needed before any paper

| Test | Why |
|---|---|
| Run the authors' MultiQT code on our streams and panels | Our re-implementation could differ |
| Step-size sweep for every method, chosen on a held-out period | Each method was run at one setting |
| The epidemic and energy datasets MultiQT uses | To compare on their ground |
| Multi-step horizons and delayed feedback | MultiQT covers delay; ours does not |
| Feedback rates 5%, 50%, 100% and unit-specific feedback | Only 20% shared feedback was run |
| More panels (the paper's HF and cross-market panels, a hierarchical retail panel) | Two panels are not enough |
| CRPS on a dense grid, and scoring that does not flatter infinite quantiles | 19 levels and clipping were used |
| Full comparison with Ochoa Rivera and Tewari and with Liu, Dobriban and Orabona | Not implemented |
| A proof for intermittent feedback in the panel setting | The identity is per revealed round; the panel statement is not written out |
