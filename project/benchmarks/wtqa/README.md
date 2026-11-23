# W-TQA for panel data: check against the paper

`WeightedTQA` implements Algorithm 1 of Tu and Giesecke, "Online conformal prediction for non-exchangeable panel data" (arXiv:2605.17705). The authors have not released code. 23 November 2026.

## 1. Same sets as a line-by-line reference

`tests/test_panel.py` holds a plain-loop transcription of Algorithm 1, one target at a time. The package returns identical thresholds (exact equality) for five settings of bandwidth, step size and finitization, with and without numba.

## 2. The paper's M5 table

M5, store CA_1, department FOODS_3, 600 conformal days from 1 January 2014, burn-in to 31 December 2013, 30 random splits, feedback revealed on 20% of rounds, alpha 0.10, h 0.6, gamma 0.01, `finite=True` (the paper's algorithm). Ours first, the paper's Table 4 in brackets (mean and standard deviation over 30 replications).

| Method | Average coverage | Tail coverage (worst 10% of units) | Relative width | Width CoV |
|---|---|---|---|---|
| Split CP | 0.899 ± 0.013 (0.904 ± 0.011) | 0.566 ± 0.035 (0.579 ± 0.030) | 1.000 (1.000) | 0.074 (0.076) |
| TQA-only | 0.928 ± 0.005 (0.931 ± 0.004) | 0.823 ± 0.012 (0.827 ± 0.011) | 0.991 (0.986) | 0.236 (0.233) |
| W-only | 0.913 ± 0.009 (0.913 ± 0.007) | 0.697 ± 0.029 (0.697 ± 0.028) | 1.002 (0.991) | 0.316 (0.332) |
| W-TQA | 0.925 ± 0.004 (0.925 ± 0.003) | 0.843 ± 0.010 (0.847 ± 0.008) | 0.974 (0.963) | 0.344 (0.358) |

Every coverage figure is within one standard deviation of the paper's. Run: `python prep.py`, then `python m5_wtqa.py 30` (115 s for all four methods, both settings of `finite`, 30 splits, on 2 cores).

Where our pipeline differs from the paper, because the paper does not give the detail or the data mirror lacks it:

| Item | Paper | Here |
|---|---|---|
| Items | 648 "after preprocessing" (rule not stated) | 656 pass our rule (on sale for at least 180 burn-in days and throughout the conformal period); 648 drawn at random per split |
| Event-day covariate | used | not available in the mirror; left out |
| Ridge penalty | not stated | 1.0 on standardized covariates |
| Profile covariates | marked in their Table 8 | log price, relative price, item age, burn-in mean, s.d., zero share, peer slope |
| Standard deviation divisor in the profile metric | not stated | n |
| Data source | Kaggle | `m5-benchmarks` public mirror (sales, SNAP, prices) |

Not checked: the synthetic panels (their parameters are not published), HF, SGSC and the cross-market panel, and the baselines TQA-B and LPCI.

## 3. `finite=False`, the package default

The paper replaces an infinite set by the largest calibration score ("finitization"). That keeps every set bounded but gives up the any-data bound on the miss rate; the paper needs its Assumption 5.6 to restore one. pyconformal keeps the infinite set by default, as it does for ACI. On M5 this costs 2.5% of sets being infinite for W-TQA; coverage is the same (tail 0.849). `output/m5_p02.txt` has both.

In `tests/test_panel.py`, targets three times harder than every peer are missed on up to 53% of revealed rounds with `finite=True`, against a bound of about 13% that holds with `finite=False`.

## 4. Speed

Against the line-by-line reference (extrapolated from 40 target-rounds), same thresholds, 2 cores:

| Rounds | Calibration units | Targets | Features | pyconformal | Reference | Ratio |
|---|---|---|---|---|---|---|
| 600 | 450 | 600 | 10 | 1.3 s | 492 s | 366x |
| 2,000 | 2,000 | 500 | 20 | 24 s | 6,093 s | 257x |

Without numba: 2.2 s on the first panel. There is no other implementation to compare with.
