# crepes 0.9.1: forensic audit

23 November 2026.

**Audited:** crepes 0.9.1 from PyPI (released 12 June 2026, the latest release), tag `v0.9.1`, commit `e930c22240756330799b1239b7dbe3235b54391a`. `base.py` and `extras.py` in the PyPI wheel are byte-identical to that commit, and `extras.py` is unchanged in `master` (commit `88a0dfa`, 8 July 2026).
**Environment:** Python 3.13, NumPy 1.26.4, scikit-learn 1.9.1, SciPy 1.17.1, pandas 3.0.6.
**Evidence:** every finding has a script here and its raw output in `output/`. Receipts are in `../CITATIONS_crepes_puncc.md`.

## Summary

The core of crepes is sound. Standard, normalized and Mondrian conformal regressors, predictive systems and classifiers met their level in all 114 coverage settings tested, p-values are uniform, the online mode agrees with a batch refit, and the test martingales keep their error rate. One fault breaks coverage, and it sits in a recipe from crepes' own notebook. Four more return a wrong or unusable result.

| # | Fault | Measured | Known to crepes? |
|---|---|---|---|
| 1 | **Out-of-bag calibration through `WrapRegressor` ignores the leave-one-out difficulty estimates of the k-nearest-neighbour estimators.** Each training object is counted as its own nearest neighbour | Coverage 83.6% at a 90% level with residual-based difficulty and k = 5; 88.1% with the default k = 25; 93.3% at a 95% level. With distance-based difficulty it over-covers instead: 95.7% to 99.7% | No report found |
| 2 | **A test object in a Mondrian category absent from calibration gets the interval [0, 0]**, with no warning | Prediction 10, interval returned [0, 0] | Yes. Issue #54, open |
| 3 | **CRPS of normalized predictive systems is wrong**: two terms are multiplied by sigma twice | One point, sigma = 5: 7.19 reported, 2.19 by definition. On a random example: 0.7% to 3% too high | Yes. Issue #40, open |
| 4 | **`MondrianCategorizer` fitted with `learner=` crashes at `apply`** with `NameError: name 'learner' is not defined` | Every call | No report found |
| 5 | **`ConformalPredictiveSystem.predict(..., seed=)` leaves NumPy's global random state reset to the seed.** The line that restores it comes after every `return` | Every call with a seed, or after `fit(seed=)` | No report found |

"No report found" means none among the 32 issue titles on the tracker on 23 November 2026.

## Fault 1: out-of-bag calibration with k-NN difficulty estimates

`DifficultyEstimator.fit(..., oob=True)` computes leave-one-out estimates for the training objects and stores them. They are returned only by `de.apply()` with no argument (`extras.py`, lines 732 to 736). `WrapRegressor.calibrate` calls `de.apply(X)` (`base.py`, line 4151). With `X` given, the k-nearest-neighbour branch looks up the neighbours of each training object in a set that contains the object itself, so the object is its own nearest neighbour at distance zero.

The effect depends on the estimator:

- **Residuals:** the object's own out-of-bag residual enters its difficulty estimate. Large residuals get large estimates, normalized scores are compressed, intervals come out too narrow.
- **Labels:** the same, more weakly.
- **Distances:** a zero distance replaces the k-th neighbour, the estimate is too small in calibration, intervals come out too wide.

crepes' notebook `docs/crepes_nb_wrap.ipynb` uses exactly this call for all three: `rf.calibrate(X_train, y_train, de=de_knn_res_oob, oob=True)`. The other notebook, `docs/crepes_nb.ipynb`, uses `de.apply()` with `ConformalRegressor` and is correct.

`c4_oob_knn.py` and `c5_oob_cause.py`: random forest, 900 training rows, 2,000 fresh test rows, 30 to 40 repetitions, 90% level.

| Difficulty estimator, `oob=True` | Through `WrapRegressor.calibrate` | With the stored estimates, `de.apply()` |
|---|---|---|
| k-NN residuals, k = 25 | **88.1%** | 89.9% |
| k-NN residuals, k = 5 | **83.6%** | 90.4% |
| k-NN labels, k = 25 | 89.8% | 90.4% |
| k-NN labels, k = 5 | **87.5%** | 90.2% |
| k-NN distances, k = 25 | 95.7% (over) | 90.1% |
| k-NN distances, k = 5 | 99.7% (over) | 89.9% |
| Forest variance | 90.1% | not affected |
| No difficulty estimate | 90.2% | not affected |

At other levels (k-NN residuals, k = 25): 93.3% at 95%, 79.0% at 80%. With 300 training rows: 88.5%.

`repro_oob_knn.py` follows the notebook line for line and prints a coverage of 83.5% at a 90% level.

Out-of-bag calibration has no finite-sample guarantee in general, and crepes does not claim one. The point here is narrower: the wrapper does not do what the documentation says it does, and the plain `ConformalRegressor` route with the same ingredients holds its level.

## Fault 2: unseen Mondrian category

`ConformalRegressor.predict_int` and `ConformalPredictiveSystem.predict` start from an array of zeros and fill it category by category, looping over the categories seen in calibration (`base.py`, lines 898 and 916 to 949). A test object in any other category is never filled. `c1_core.py`, part (c): prediction 10, category seen in calibration: [8.53, 11.47]; category not seen: [0, 0]. No warning or error. The online methods handle the same case correctly and return an infinite interval.

Reported in issue #54, open.

## Fault 3: CRPS with difficulty estimates

`get_crps` (`base.py`, lines 2841 to 2842) multiplies the two terms next to the outcome by `sigma`, although `y - cpd[...]` is already on the scale of `y`. Standard predictive systems (sigma = 1) are unaffected: crepes reports 0.63679 and the definition gives 0.63679. Normalized: 0.60511 against 0.60085 on the distributions crepes itself returns (`c2_more.py`, part (h)). The size of the error grows with sigma, so the metric penalises normalized systems and can reverse a comparison between a normalized and a standard one.

Reported in issue #40, open.

## Fault 4: `MondrianCategorizer` with a learner

`extras.py`, line 413: `scores = learner.predict(X)` should read `self.learner`. `MondrianCategorizer().fit(X_cal, learner=model)` succeeds; `apply` then raises `NameError`. The `oob=True` path and the `f=` path work. crepes' documentation uses `f=model.predict`, which avoids the line.

## Fault 5: the global random state

`ConformalPredictiveSystem.predict` seeds NumPy's global generator when `seed` is given, and restores the saved state at line 2597, after the last of three `return` statements. The restore never runs. After `cps.predict(..., seed=0)`, or after any `predict` on a system fitted with `fit(seed=0)`, the caller's next `np.random` draw is the first draw of seed 0. A bootstrap or simulation loop that calls it repeats the same random numbers on every pass. `ConformalRegressor.predict_p` and `ConformalClassifier.predict_set` restore the state correctly (`c1_core.py`, part (d)).

## Smaller findings

| Finding | Evidence |
|---|---|
| The rank is computed in floating point. At 90% and 80% levels it is one rank too high whenever `n + 1` is a multiple of 10 or 5: 199 of 1,992 and 399 of 1,997 calibration sizes. At n = 9 (90%) and n = 4 (80%) it returns an infinite interval where the largest residual is valid. Intervals are wider than needed; coverage is not harmed. Levels 0.95, 0.975 and 0.99 are exact | `c1_core.py`, part (b) |
| The same in predictive-system intervals: wider than exact in 100 of 1,982 sizes at 90% and 200 of 1,992 at 80%; never narrower | `c1_core.py`, part (b) |
| Online and batch intervals of the Mondrian predictive system differ at some sizes (4 of 80 steps in one run). Both are valid; the online one is wider | `c3_online_mondrian.py` |
| `WrapRegressor.fit` and `WrapClassifier.fit` return `None`, so `WrapRegressor(m).fit(X, y).calibrate(...)` fails, while `calibrate` does return the object | `c6_rest.py` |
| `binning` puts equal values in different bins when values are tied | `c2_more.py`, part (j) |

## What passed

| Area | Test | Result |
|---|---|---|
| Coverage on i.i.d. data (`c1_core.py`) | Regressor and predictive system: standard, normalized, Mondrian, both; 19 to 1,000 calibration points; levels 0.9, 0.95, 0.99. Classifier: standard, Mondrian, class-conditional; smoothing on and off | 114 settings, none under its level |
| p-values (`c2_more.py`, `c6_rest.py`) | Regressor, predictive system, classifier, batch and online, continuous and heavily tied scores | Uniform: P(p < 0.1) between 0.098 and 0.100 |
| Online mode | Regressor and predictive system against a batch refit at every seventh step | Identical, apart from the Mondrian predictive system noted above |
| Online classification | Standard, Mondrian, no warm start | 89.9% to 90.0% at 90% |
| Wrappers with a random forest (`c2g.py`, `c4_oob_knn.py`) | Split calibration with five difficulty estimators and a predictive system; out-of-bag calibration plain and with forest variance | 89.4% to 90.3% at 90% |
| Classifier wrapper | Hinge, margin, class-conditional, out-of-bag, no smoothing | 90.1% to 90.7%; class-conditional gives 89.4% to 90.9% inside each class |
| Percentiles of the predictive system | 5/95, 10/90, 25/75, 50 | Within noise of the stated levels |
| `evaluate` | Error, mean and median width, set sizes, singleton and empty rates against hand calculations | All agree. CRPS agrees for standard systems |
| Martingales (`c2_more.py`, part (i)) | SimpleJumper, SleeperStayer, SleeperDrifter, a composite; 600 runs of 1,000 uniform p-values | Reached 20 in 1.7% to 4.2% of runs (5% allowed); all detected a change in every run |
| pandas | Frames with shuffled indexes, Series, lists, string labels | Same results as NumPy |

## Coverage of this audit

| Part | Status |
|---|---|
| `base.py`: `ConformalRegressor.fit`, `predict_int`; `ConformalPredictiveSystem.fit`, `predict`, `predict_int`, `predict_int_online`; `p_values_batch`; CRPS; both `calibrate` methods; `ConformalClassifier.predict_set` | Code read and tested |
| `base.py`: the other online functions, `predict_percentiles`, `predict_cpds`, `evaluate` | Tested. Code not read line by line |
| `extras.py`: `DifficultyEstimator`, `MondrianCategorizer` | Code read and tested |
| `extras.py`: `hinge`, `margin`, `binning` | Tested |
| `martingales.py` | Tested for error rate and detection. Code not read; the betting functions were not tested separately |
| Class-conditional online p-values (`p_values_online_class_cond`), `semi_online_p_values(optimized=True)` | Not tested |

## How to reproduce

    pip install crepes==0.9.1 scikit-learn pandas scipy
    python repro_oob_knn.py        # fault 1, 1 minute
    python c1_core.py 300          # coverage, rank, faults 2 and 5; 8 minutes
    python c2_more.py 300          # p-values, online, metrics (fault 3), martingales
    python c4_oob_knn.py 40        # fault 1 in full
    python c5_oob_cause.py 30      # its cause
    python c6_rest.py              # fault 4 and the rest
