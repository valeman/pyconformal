# FAQ

## Does quantile regression in pyconformal rely on CatBoost?

No. The only required dependency is NumPy.

- **CQR, CIR and CTI take quantile predictions from any model.** Pass them as an array: `cqr.calibrate(quantiles=q_cal, y=y_cal)`. See [CQR](methods/cqr.md).
- **`QuantileForest` is built in.** It needs numba (`pip install "pyconformal[fast]"`), nothing else. See [Quantile forest](methods/quantile-forest.md).
- **`QuantileBooster` is an optional wrapper** for CatBoost, XGBoost or LightGBM. Only the library you name has to be installed. Its default is `"catboost"`, so write `QuantileBooster("lightgbm")` or `QuantileBooster("xgboost")` to use another. See [Quantile models from boosting libraries](methods/quantile-booster.md).

The README's first example uses CatBoost as the model, which is where the question comes from. Any model works there.

## What has to be installed for what?

| You use | Install |
|---|---|
| Split conformal, Mondrian, weighted, classification scores, Venn-Abers, ACI and the other online methods, multi-step, predictive systems | `pip install pyconformal` (NumPy only) |
| The fast compiled paths and `QuantileForest` | `pip install "pyconformal[fast]"` (adds numba) |
| `QuantileBooster("catboost")` | `pip install "pyconformal[catboost]"` |
| `QuantileBooster("xgboost")` | `pip install "pyconformal[xgboost]"` |
| `QuantileBooster("lightgbm")` | `pip install "pyconformal[lightgbm]"` |

scikit-learn is not required. The examples on this site use it only to make data and fit a model.

## Can I do Mondrian conformal prediction?

Yes. Pass `groups=` to `calibrate` and again to the prediction call. See [Mondrian regression](methods/mondrian.md) and [Mondrian classification](methods/mondrian-classification.md). For one threshold per class, use `SplitConformalClassifier(class_conditional=True)`.

## Does my model have to be a scikit-learn model?

No. Either pass any object with `predict` (or `predict_proba` and `classes_` for classification), or skip the model and pass predictions: `calibrate(y_pred=..., y=...)`, `calibrate(proba=..., y=...)`, `calibrate(quantiles=..., y=...)`.

## How big should the calibration set be?

The guarantee holds at any size. For a level `alpha` you need at least `1/alpha - 1` calibration points for a finite interval (9 for 90%, 19 for 95%, 99 for 99%); below that the interval is infinite, which is the honest answer. A few hundred to a thousand points give stable widths. With Mondrian, this applies inside each category.

## Is the guarantee per prediction?

No. Split conformal coverage is on average over calibration and test data. For coverage inside categories use [Mondrian](methods/mondrian.md); for intervals that adapt to difficulty use [normalized intervals](methods/normalized.md) or [CQR](methods/cqr.md).

## My data is a time series. Which method?

Start with [ACI](methods/aci.md): it needs only forecasts and actuals and holds its long-run miss rate under drift. If you do not want to choose its step size, use [DtACI or AgACI](methods/dtaci-agaci-pid-nexcp.md). For several steps ahead, see [multi-step forecasts](methods/multi-step.md).

## Why can an interval be infinite?

Because that is what the theory requires in two cases: too few calibration points for the level, and adaptive methods whose working level has reached zero. pyconformal returns the infinite interval. It does not clip it, because clipping breaks the guarantee.

## A method I need is missing

See the list at the bottom of [What's inside](whats-inside.md) and open an [issue](https://github.com/valeman/pyconformal/issues).
