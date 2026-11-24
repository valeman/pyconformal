# conformal-tights: first-pass review

24 November 2026. Package: https://github.com/superlinear-ai/conformal-tights, version 0.5.0, commit `d8ab41df3fc877e94f839f4b18947b503e5edea1` (21 March 2026), MIT. 882 lines; one method (`ConformalCoherentQuantileRegressor`) and a Darts wrapper. All source files were read. Tests ran on xgboost 3.4.1, scikit-learn 1.9.1.

This is a first pass, not a full audit: one synthetic data model, see "Not done".

## What the method does

Fits a point model and an XGBoost quantile model (5 levels) on the training part. On a first calibration part it fits a linear quantile regression, constrained not to cross, from the 5 XGBoost quantiles to the residual. On a second calibration part it shifts each quantile line by a constant. It does this for absolute and for relative residuals and, for each test point, returns whichever of the two is narrower.

## Findings

| # | Finding | Where | Evidence |
|---|---|---|---|
| 1 | No calibration step below about 1,423 training rows. The second calibration part is 30% of the calibration set (itself 30% of the data, at most 1,440); with fewer than 128 rows there the shift is set to zero. The interval is then a quantile regression fitted on the calibration set, with no conformal correction. | `_conformal_coherent_quantile_regressor.py` lines 121-131, 242 | Target 0.95: coverage 0.866 at n=300, 0.914 at n=1,000. Target 0.90: 0.824 and 0.867. 100 trials each; 96-100% of trials below target. |
| 2 | The shift is a plain empirical quantile, without the finite-sample correction, and ignores sample weights (marked TODO in the code). | line 255 | Read from code. |
| 3 | For each test point the narrower of two calibrated intervals is returned. Picking the narrower of two intervals point by point lowers coverage. | lines 304-309 | n=5,000, target 0.95: absolute only 0.953, relative only 0.954, returned 0.947. Target 0.90: 0.906, 0.909, 0.897. |
| 4 | For quantiles between the lowest and highest, the shift is clipped to almost nothing, to keep the lines from crossing on the calibration points. | `_coherent_linear_quantile_regressor.py` `intercept_clip`; line 256 | 5 levels, n=5,000, 6 fits: the data asked for shifts of 0.06-0.09 at levels 0.25-0.75; applied 0.000-0.008. |
| 5 | Slow: the non-crossing fit is a linear program solved at prediction time. | `coherent_linear_quantile_regression` | First `predict_quantiles` call with 5 levels, n=5,000: 41-44 s on 2 cores. |
| 6 | Quantiles can still cross on new points (the constraint holds only on calibration points). | same | 0.16% of test points, 5 levels. |
| 7 | `ŷ.astype(self.y_dtype_)` discards its result. | `_coherent_linear_quantile_regressor.py` line 247 | Read from code; no effect measured. |

With enough data (n=5,000) coverage is close to target: 0.947 at 0.95 and 0.897 at 0.90, and quantile frequencies at five levels were within 0.01 of the level. The method has no finite-sample coverage guarantee and the package's documentation does not state one.

## Anything pyconformal lacks?

No method. The useful ideas are conveniences: a Darts wrapper, a zero-argument default (`"auto"`), targets of datetime type.

## Not done

Real datasets; XGBoost defaults (60 trees of depth 3 were used for speed); sample weights; the Darts forecaster; a side-by-side run against pyconformal on the same data; finding 4 on more than 6 fits.

Run: `python e1_coverage.py 100`, `python e2_call_order.py`, `python e3_bias_clip.py`.
