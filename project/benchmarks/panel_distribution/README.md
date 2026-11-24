# Predictive distributions for panels: `PanelPredictiveSystem`

24 November 2026. Each target unit gets a full predictive distribution every round: the weighted distribution of its peers' signed errors (W-TQA's similarity weights, arXiv:2605.17705), with its quantiles adjusted online by MultiQT (arXiv:2512.23671) on the rounds where the target's own outcome is revealed.

A search of arXiv (about 60 queries, 24 November 2026) found no method that gives predictive distributions for panel data with late or missing feedback. The parts are published; the combination, its use under partial feedback and the evidence below are ours. See `../adaptive_cpd/LITERATURE.md`.

## Results

Panels, pipelines, splits and feedback (20% of rounds) are those of `../wtqa`. 30 random splits, 19 levels (0.05 to 0.95). "Base only" is the weighted cross-sectional distribution without adjustment; with equal weights it is a plain conformal predictive system on the cross-section.

| Panel | Variant | Per-unit calibration error | Worst 10% of units | Pinball loss | Worst 10% of units | Central 90% coverage, worst 10% of units | 90% width |
|---|---|---|---|---|---|---|---|
| M5 | equal weights, base only | 0.123 | 0.253 | 0.2156 | 0.4125 | 0.571 | 2.365 |
| M5 | similarity weights, base only | 0.116 | 0.239 | 0.2071 | 0.3766 | 0.724 | 2.313 |
| M5 | equal weights, adjusted | 0.031 | 0.061 | 0.1828 | 0.3166 | 0.869 | 2.167 |
| M5 | similarity weights, adjusted | 0.030 | 0.059 | 0.1804 | 0.3035 | 0.881 | 2.093 |
| SGSC | equal weights, base only | 0.109 | 0.244 | 0.0411 | 0.0910 | 0.606 | 0.519 |
| SGSC | similarity weights, base only | 0.103 | 0.244 | 0.0416 | 0.0936 | 0.720 | 0.595 |
| SGSC | equal weights, adjusted | 0.027 | 0.051 | 0.0392 | 0.0770 | 0.851 | 0.494 |
| SGSC | similarity weights, adjusted | 0.026 | 0.051 | 0.0393 | 0.0769 | 0.860 | 0.524 |

Per-unit calibration error: mean over the 19 levels of |observed frequency - level|, per target unit.

Against a plain cross-sectional predictive system, the full method cuts per-unit calibration error by a factor of four on both panels, pinball loss by 16% on M5 and 4% on SGSC, and narrows the central 90% band on M5 by 12% while raising coverage of the worst-served units from 0.57 to 0.88. Most of the gain is the online adjustment; the similarity weights add about 1% in pinball loss on M5 and nothing on SGSC.

With the adaptive step's floor lowered from 0.1 to 0.02 (SGSC errors are small): pinball 0.0377, calibration error 0.029. The floor is in outcome units and should be set to the scale of the errors.

Comparison with other online recalibrators on the same base (10 splits): `../adaptive_cpd/LITERATURE.md`.

Speed: 1.6 s for one M5 split (600 rounds, 454 calibration units, 194 targets, 19 levels), 0.9 s for SGSC, on 2 cores.

Run: `python run.py m5 30`, `python run.py sgsc 30` (needs the data built by the scripts in `../wtqa`).

## Not done

Other feedback rates and unit-specific feedback; multi-step horizons; more panels; CRPS on a dense grid; a written-out proof for partial feedback (the docstring states what follows from MultiQT's theorem applied to the revealed rounds); comparison with the two other multi-level methods (arXiv:2605.12668, arXiv:2611.12059).
