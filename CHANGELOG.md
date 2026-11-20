# Changelog

## 0.1.0 (unreleased)

First working release. Version 0.0.1 only reserved the name.

One scheme for every method: `fit` trains the model, `calibrate` uses held-out
data, `predict_*` gives the output, `update` takes one new observation. Every
method works from model outputs or from a model.

**Regression**
- Split conformal regression, with normalized (`sigma=`), Mondrian
  (`groups=`) and weighted (`weights=`) variants
- Jackknife+ and CV+ (`CrossConformalRegressor`)
- Conformalized quantile regression (CQR), conformal thresholded intervals
  (CTI), conformal interquantile regression (CIR and CIR+)
- Conformal predictive systems: standard, normalized and Mondrian, with exact
  CRPS
- `QuantileForest`: a parallel quantile regression forest, with out-of-bag
  calibration

**Classification**
- Split conformal classification with LAC, APS, RAPS and SAPS scores and
  class-conditional sets
- Venn-Abers calibration, binary and multiclass, and cross Venn-Abers

**Time series**
- Adaptive conformal inference (ACI), DtACI, AgACI, NexCP, conformal PID
- EnbPI and SPCI
- Conformal seasonal pools (CSP)
- STL and MSTL decomposition

**Other**
- Optional numba kernels (`pip install pyconformal[fast]`); results are
  identical with and without numba where both paths exist. `QuantileForest`
  and `SPCI` need numba.
