# Changelog

## 0.1.1 (22 November 2026)

Fixes from an independent forensic audit of 0.1.0. Upgrade if you use any of
the methods below; every fix has a regression test built from the audit's
counterexample.

**Coverage and correctness**
- ACI, DtACI, AgACI, NexCP and conformal PID: the reported coverage now always
  equals the share of outcomes inside the issued intervals. An empty
  prediction set was returned as a single point, which could contain the
  outcome while being counted as a miss; it is now `(nan, nan)`.
- Conformal PID: on a constant series the step size became zero and the
  threshold stuck below zero. When the recent scores are all equal a nonzero
  step is kept. Elsewhere results are unchanged.
- EnbPI: residuals of rows that every member had seen were computed from
  in-bag predictions. Only out-of-bag residuals are used now; `fit` trains
  extra members when needed. With two bootstraps coverage at 90% rose from
  84% to 90%. SPCI refuses member predictions without an out-of-bag value.
- EnbPI with `batch_size > 1`: splitting a stream into several `run` calls
  now gives the same intervals as one call.
- Conformal predictive systems: CRPS lost precision when all values shared a
  large offset and could turn negative; it is now translation invariant. A
  NaN threshold in `predict_cdf` gives NaN instead of a tail probability.
- Weighted split conformal: the mass boundary was lowered by a tolerance; it
  is now never lowered. Equal weights give exactly the unweighted interval;
  all-zero weights are rejected; very large weights no longer overflow.
- ACI, DtACI, AgACI and NexCP at the configured level use the exact decimal
  rank, as split conformal does; a tolerance gave one rank too low at
  boundary levels. Levels reached by the adaptive updates use a stated
  floating-point rule.
- Refitting a model through `fit` clears the old calibration, so a new model
  can no longer be used with the previous model's residuals.
  `set_params` checks new values before changing anything and keeps the
  calibration only when `alpha` alone changes.
- Classification: probabilities are checked (finite, in [0, 1], rows summing
  to 1, the calibrated number of columns); distinct classes are required.
- Venn-Abers with Boolean labels now honours the order of `classes`.
- `sample_weight` and other per-row fit parameters are resampled with the
  rows in EnbPI, jackknife+/CV+ and cross Venn-Abers.
- DtACI: `update` straight after `calibrate` raised `TypeError`.
- CIR: `window="shortest"` (the authors' rule) has no coverage guarantee and
  now warns; on a simple exchangeable example it covers 0% at a 90% target.
- AcMCP: documented that the default `Csat` depends on the length of the
  replayed history, as in the R package; pass `Csat` to avoid that.

**Speed**
- Classification thresholds are computed once per level instead of on every
  prediction.
- Predictive-system quantiles no longer copy all calibration scores on every
  call.

**Testing**
- The test suite runs with warnings as errors.

## 0.1.0 (22 November 2026)

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
- Multi-step split conformal prediction and AcMCP
- Conformal seasonal pools (CSP)
- STL and MSTL decomposition

**Other**
- Optional numba kernels (`pip install pyconformal[fast]`); results are
  identical with and without numba where both paths exist. `QuantileForest`
  and `SPCI` need numba.
