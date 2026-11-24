# MultiQT: check against the authors' code

`pyconformal.MultiQT` implements Procedure 1 of Ding, Gibbs and Tibshirani, "Calibrated multi-level quantile forecasting" (arXiv:2512.23671), with the learning-rate rule of their Section 5. 24 November 2026.

The authors' code is at https://github.com/tiffanyding/multiQT (commit `3300f75c5cedb7f888af4058032a94940b682ccb`, 1 August 2026). It carries no licence file, so none of it is copied here: ours is written from the paper, and theirs is only run beside it. To repeat the check, clone their repository into this folder and run the two scripts.

| Check | Result |
|---|---|
| Same forecasts as the authors' `multiQT` (`example.py`) over 12 settings: 3 and 7 levels, quantile and zero base forecasts, fixed and adaptive step, delays 0, 1 and 3 | Largest absolute difference 8.9e-16 |
| Speed, one series, 19 levels, 2,000 rounds | 0.09 s against 1.4 s: 15x |
| Speed, one series, 20,000 rounds | 1.0 s against 14.6 s (theirs extrapolated from 2,000 rounds): 15x |
| 500 series side by side, 600 rounds | 2.8 s in one call. Their code handles one series per call; at the measured 0.7 ms a round that is about 210 s. |

Additions over the published code: many series in one call, outcomes that are never revealed (`observed=`), an adjustable floor for the adaptive step (`lr_floor`), and a result object with calibration, pinball loss, intervals and a CDF.
