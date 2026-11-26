# Weighted conformal predictive systems: check against crepes-weighted

`compare_crepes_weighted.py` compares `ConformalPredictiveSystem(..., weights=)` with crepes-weighted 0.1.3, the code released with Jonkers, Van Wallendael, Duchateau and Van Hoecke (2024), and checks both against the unweighted system and against the guarantee. Output of 26 November 2026: `output.txt`.

| Check | Result |
|---|---|
| Distribution function Q(y, tau), same random tau, 50 problems, half with ties | The two agree to 9e-16. |
| Equal weights, 99 calibration points, 5% and 95% quantiles | crepes (unweighted) uses ranks 5 and 95. pyconformal with equal weights: 5 and 95. crepes-weighted with equal weights: 3 and 96. |
| Same, 2.5% and 97.5% | crepes 2 and 98; pyconformal 2 and 98; crepes-weighted 1 and no finite upper end. |
| Known covariate shift, true likelihood-ratio weights, 90% interval, 200 calibration points | pyconformal covers 92.2% with 3.2% infinite intervals; crepes-weighted covers 93.9% with 26.1% infinite. Both miss at most 5% on each side at every size tested (20, 50, 200, 1,000). |
| Same, 1,000 calibration points | 90.2% and 90.5%. |

Reading. The distribution functions are the same. The quantile rules differ: crepes-weighted takes ranks that are one to two places further out, so with equal weights it does not return the quantiles of the unweighted system, and it returns infinite intervals more often. Both rules are valid in the simulation. pyconformal's rule is the one that follows from the weighted quantile lemma of Tibshirani et al. (2019) applied to each tail, and it reduces exactly to the unweighted rule (`floor(q (n + 1))` below, `ceil(q (n + 1))` above).

One thing to know when comparing tie handling: crepes-weighted tests `y_pred + score == y`, pyconformal tests `(y - y_pred) / sigma == score`. With non-zero predictions rounding can make these differ, so the tie cases above use predictions of zero.
