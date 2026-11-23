# Predictive distributions that stay calibrated under drift: research prototype

23 November 2026. Code in this folder is a prototype, not part of the package API. Nothing here has been peer reviewed, and the literature check behind "new" was six searches, not a full review.

## The idea

A conformal predictive system gives a whole predictive distribution, valid when the data are exchangeable. Time series are not. `PITTracker` (`adaptive_cpd.py`) repairs any predictive distribution online:

1. When an outcome arrives, compute its probability integral transform (PIT): where it fell in the distribution that was issued for it.
2. For each probability level tau_k keep a threshold theta_k on the PIT scale and move it by `theta_k += gamma * (tau_k - 1{PIT <= theta_k})`. No clipping.
3. The recalibrated tau_k-quantile is the base distribution's theta_k-quantile; minus infinity if theta_k <= 0, plus infinity if theta_k >= 1.

This is the adaptive conformal update of Gibbs and Candès run at every level at once, on the PIT scale so that one tracker serves any base distribution.

## What can be proved

**1. Every level is calibrated in the long run, for any sequence.** If theta_k < 0 the indicator is 0 and theta_k rises by gamma tau_k; if theta_k >= 1 it is 1 and theta_k falls by gamma (1 - tau_k). So theta_k stays in [-gamma (1 - tau_k), 1 + gamma tau_k]. Summing the updates over n revealed outcomes, `freq{PIT <= theta_k} - tau_k = (theta_k,1 - theta_k,n+1) / (gamma n)`, hence

    |freq{y_t <= q_k,t} - tau_k| <= (1 + gamma) / (gamma n)        for every level k.

No assumption on the data, the forecaster or the base distribution (ties aside).

**2. Quantiles can cross, but by less than gamma, and a crossing does not persist.** Take levels tau_j < tau_k with theta_j <= theta_k. After one update the gap theta_k - theta_j changes by gamma (tau_k - tau_j) if the PIT is on the same side of both, and by gamma (tau_k - tau_j) - gamma if it falls between them. So the gap can become negative only when it was below gamma, and never below -gamma (1 - (tau_k - tau_j)). Once crossed, the next update either widens the gap by gamma (tau_k - tau_j) or, if the PIT falls between the two, by gamma (tau_k - tau_j) + gamma, which uncrosses them.

`mode="sorted"` returns the thresholds in order, so the output is always a valid distribution. Each sorted threshold is within gamma of a threshold that has guarantee 1. **Not proved:** the exact bound for the sorted thresholds themselves. In every experiment below sorted and unsorted calibration agree to the third decimal, except on random-walk errors.

## Option 1: one series (`single_series.py`)

Base: a rolling conformal predictive system on the last 200 forecast errors. 19 levels, 3,000 steps, 5 seeds. "Worst level" is the largest calibration error over the 19 levels.

| Stream | Base: worst level | Adaptive, gamma 0.02: worst level | Base: pinball | Adaptive: pinball |
|---|---|---|---|---|
| i.i.d. errors | 0.005 | 0.001 | 0.2977 | 0.3010 |
| Noise x5 from the middle | 0.012 | 0.001 | 0.8985 | 0.9050 |
| Noise grows 0.1% a step | 0.027 | 0.001 | 2.3168 | 2.3443 |
| Forecast bias drifts | 0.083 | 0.002 | 0.3036 | 0.3015 |
| Random-walk error | 0.182 | 0.013 | 0.2455 | 0.1207 |
| Regime switch every 250 steps | 0.022 | 0.002 | 0.8034 | 0.7765 |
| Skew flips | 0.011 | 0.002 | 0.2614 | 0.2636 |
| M5 store total, seasonal-naive forecast | 0.016 | 0.003 | 101.0 | 102.1 |
| SGSC mean load, same-slot-yesterday forecast | 0.020 | 0.001 | 0.0223 | 0.0184 |

The bound from guarantee 1 at gamma 0.02 and 3,000 steps is 0.017; the measured errors are far inside it. Calibration is repaired on every stream. Sharpness (pinball loss) improves where drift is strong and costs about 1% where there is none. At gamma 0.02, 0.1% to 4.7% of quantiles are infinite (scored at the widest finite value). Full table with three step sizes: `output/single_series.txt`.

## Option 2: a panel (`panel_cpd.py`)

Base: the weighted distribution of the peers' signed errors in the same round, with W-TQA's similarity weights or equal weights. The tracker is fed only on the 20% of rounds where the target's outcome is revealed. Same panels, pipelines and settings as `../wtqa`; 30 splits.

| Panel | Distribution | Per-unit calibration error | Worst 10% of units | Pinball | Worst 10% of units | Central 90% coverage, worst 10% of units |
|---|---|---|---|---|---|---|
| M5 | equal weights, fixed levels (plain cross-sectional CPS) | 0.123 | 0.253 | 0.2156 | 0.4125 | 0.571 |
| M5 | similarity weights, fixed levels | 0.116 | 0.239 | 0.2071 | 0.3766 | 0.732 |
| M5 | equal weights, adaptive levels | 0.068 | 0.138 | 0.1964 | 0.3594 | 0.865 |
| M5 | similarity weights, adaptive levels | 0.066 | 0.134 | 0.1916 | 0.3373 | 0.886 |
| SGSC | equal weights, fixed levels | 0.109 | 0.244 | 0.0411 | 0.0910 | 0.606 |
| SGSC | similarity weights, fixed levels | 0.103 | 0.244 | 0.0416 | 0.0936 | 0.721 |
| SGSC | equal weights, adaptive levels | 0.061 | 0.127 | 0.0383 | 0.0803 | 0.844 |
| SGSC | similarity weights, adaptive levels | 0.058 | 0.129 | 0.0387 | 0.0824 | 0.867 |

Per-unit calibration error is the mean over 19 levels of |observed frequency - level|, per unit. Adapting the levels roughly halves it on both panels and cuts pinball loss by 7% to 11%. Similarity weights add a further 2% to 3% on M5 and nothing on SGSC pinball. The cost: pooled over all units the adaptive distributions are too wide in the middle (central 90% covers 93% to 95%).

## Limits

- Guarantee 1 is long-run and marginal over time. It says nothing about any single step, and nothing can without assumptions.
- The sorted-threshold bound is not proved.
- Infinite quantiles are part of the method, as in ACI. They are scored at the widest finite value here, which flatters pinball loss slightly.
- No comparison yet with online recalibration from the calibration literature (Deshpande, Marx and Kuleshov; Marx, Kuleshov and Ermon), which is the closest prior work, nor with Vovk's protected probabilistic regression. Novelty is not established until that is done.
- Two panels, nine streams, one base distribution each. No multi-step horizons.
