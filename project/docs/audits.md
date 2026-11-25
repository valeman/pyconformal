# Audits of other libraries

A conformal method makes one promise: a 90% interval covers at least 90% of the time, for any model, at any sample size. Between 20 and 25 November 2026 we tested the current releases of fourteen packages on plain i.i.d. data, where that promise must hold, and then went through five conformal libraries module by module: MAPIE 1.6.0, crepes 0.9.1, PUNCC 0.9.3, TorchCP 1.2.1 and conformal-tights 0.5.0.

Each fault was traced to a line of source, reproduced by a script that uses only the package concerned, and checked against the package's development branch. Each report also lists what passed and what was not covered. All findings are for the versions named, on the dates named; later releases may differ.

**Coverage lost, with no error and no warning.**

| Package | Setting | Target | Measured |
|---|---|---|---|
| TorchCP 1.2.1 | `SAPS(randomized=False)`, an accurate classifier | 90% coverage | **0% to 36%** |
| PUNCC 0.9.3 | box-wise object detection, detector's boxes too small | 90% of true boxes inside | **0%** |
| TorchCP 1.2.1 | AgACI on drifting series | 10% misses | **27% to 45% misses** |
| PUNCC 0.9.3 | APS and RAPS, one row per `predict` call | 90% coverage | **45%** (68% with two rows, 89% with five) |
| TorchCP 1.2.1 | CQR-R | 90% coverage | **57.6%**: the interval is the raw quantile band |
| statsforecast 2.1.1, mlforecast 1.1.0 | default (2 calibration windows) | 90% coverage | **63% to 65%** |
| MAPIE 1.6.0 | ACI on drifting series | 10% misses | **14% to 33% misses** |
| TorchCP 1.2.1 | ACI with random-walk errors | 10% misses | **14.0% misses** |
| Darts 0.48.1, conformalpy 1.0.0 | 3 calibration points | 90% coverage | **75%** |
| conformal-tights 0.5.0 | defaults, 300 rows, eight public datasets | 90% coverage | **80.6%** (84.2% with 1,000 rows, 87.2% with 5,000) |
| MAPIE 1.6.0 | APS and RAPS with `include_last_label=False` | 90% coverage | **81.5% and 83.6%** |
| crepes 0.9.1 | out-of-bag calibration with k-nearest-neighbour difficulty, as in its notebook | 90% coverage | **83.6%** with k = 5, 88.1% with the default k = 25 |
| TorchCP 1.2.1 | NorABS used as its docstring says | 90% coverage | **83.6%** |
| MAPIE 1.6.0 | default RCPS confidence bound | wrong in at most 10% of runs | **up to 19.7%** |
| TorchCP 1.2.1 | `ConformalLM` with a scaler | risk above its limit in at most 5% of draws | **25.5%** |

On the same tests pyconformal behaved as the theory says: its ACI missed 10.1%
on the drifting series, split conformal covered 90.0% on the eight datasets
with 300 rows, and its APS, RAPS and SAPS sets covered 91% whether rows were predicted
together or one at a time.

**Wrong or unusable results.**

| Package | What happens |
|---|---|
| crepes 0.9.1 | A test object in a Mondrian category that was not in calibration gets the interval [0, 0], with no warning |
| crepes 0.9.1 | CRPS with difficulty estimates: 7.19 reported where 2.19 is correct |
| crepes 0.9.1 | `MondrianCategorizer` fitted with `learner=` raises `NameError`; `predict(seed=)` leaves NumPy's global random state reset |
| TorchCP 1.2.1 | NorABS raises an error on any batch of more than one row |
| TorchCP 1.2.1 | `ClusteredPredictor` returns every label for classes without a cluster; `WeightedPredictor` sorts weights separately from scores and returns sets 41% to 62% larger than correct weighting |
| MAPIE 1.6.0 | The default Holm procedure in learn-then-test returns no threshold in 161 of 300 runs where Bonferroni returns one in all 300 |
| PUNCC 0.9.3 | EnbPI crashes when residuals are updated after every observation |
| conformal-tights 0.5.0 | Sample weights are ignored in its calibration step: with corrupted rows given weight zero the interval is three times wider than it should be |

**Two patterns are worth knowing.** MAPIE and TorchCP both clip the working
level of ACI. Adaptive conformal inference is defined without a clip, and its
guarantee, validity under any drift, needs the level to be free to leave
[0, 1]. On stationary data the clip never binds, so ordinary tests pass; on
drifting data the guarantee is gone. And crepes and TorchCP both let a
calibration object be its own nearest neighbour when estimating difficulty,
which gives the same 83.6% in both.

**What was correct.** The basic split-conformal regression and classification
routines of MAPIE, crepes, PUNCC and TorchCP were exact in every test, and
PUNCC's regression side was the cleanest we examined. The faults sit in
options, newer modules and time-series methods. With two calibration errors no
finite 90% interval is valid; at least nine are needed, and the only honest
answer below that is an infinite interval. The statsforecast behaviour is
[issue #1349](https://github.com/Nixtla/statsforecast/issues/1349). Some items
were already on the projects' trackers (MAPIE #1021 and #998, crepes #40 and
#54, TorchCP #126); the reports say which.

Full reports, with scripts and raw output:
[MAPIE](https://github.com/valeman/pyconformal/blob/main/project/benchmarks/mapie_audit/README.md) ·
[crepes](https://github.com/valeman/pyconformal/blob/main/project/benchmarks/crepes_audit/README.md) ·
[PUNCC](https://github.com/valeman/pyconformal/blob/main/project/benchmarks/puncc_audit/README.md) ·
[TorchCP](https://github.com/valeman/pyconformal/blob/main/project/benchmarks/torchcp_audit/README.md) ·
[conformal-tights](https://github.com/valeman/pyconformal/blob/main/project/benchmarks/conformal_tights_audit/README.md)

## Why this matters for pyconformal

These faults are the reason for three rules in pyconformal: every threshold comes from one exact rank routine; calibration and prediction always go through the same scoring code; and a method is released only after it has been compared with its definition, with reference code where it exists, and with its guarantee on repeated draws. See the [FAQ](faq.md).
