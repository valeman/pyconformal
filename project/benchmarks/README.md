# Benchmarks

Scripts behind every number on the front page. Run them in a separate
environment so the audited packages do not mix with your own:

```bash
conda create -n cp-audit python=3.12 -y && conda activate cp-audit
pip install "pyconformal[fast]" statsforecast mlforecast darts mapie venn-abers crepes puncc scikit-learn
python nixtla_coverage.py
```

| Script | What it measures |
|---|---|
| `nixtla_coverage.py` | Coverage of statsforecast and mlforecast conformal intervals |
| `darts_coverage.py` | Coverage of Darts `ConformalNaiveModel` with short calibration |
| `mapie_audit.py` | Coverage of MAPIE across 84 configurations (about 25 minutes) |
| `speed_venn_abers.py` | Venn-Abers speed and agreement: pyconformal, venn-abers, MAPIE |
| `speed_aci.py` | ACI speed and coverage: pyconformal against MAPIE |
| `speed_csp.py` | CSP speed: pyconformal against csp-forecaster 0.2.0 |
| `speed_predictive.py` | Conformal predictive systems: pyconformal against crepes (several minutes) |
| `crepes_crps_check.py` | crepes CRPS for normalized systems against the definition |
| `speed_classification.py` | Prediction sets: coverage, size and time against TorchCP, MAPIE and PUNCC (needs PyTorch) |
| `torchcp_audit.py` | TorchCP threshold exactness and coverage, including SAPS |
| `speed_split.py` | Split conformal regression: pyconformal, MAPIE, crepes, PUNCC |
| `speed_quantile.py` | CQR against MAPIE and PUNCC; CTI against its authors' code; CIR widths |
| `speed_cir.py` | CIR and CIR+ against the authors' code (pass the path of a clone of github.com/orince/CIR; needs torch and torchcp) |
| `speed_forest.py` | Quantile forest against quantile-forest (Zillow): fit, predict, accuracy |
| `speed_mstl.py` | MSTL against statsforecast and statsmodels, with output differences |
| `speed_cross.py` | CV+ against MAPIE |
| `speed_multistep.py` | Multi-step split conformal and AcMCP against R's conformalForecast (pass the path to `Rscript`) |
| `speed_timeseries.py` | ACI, DtACI, AgACI, PID, NexCP, EnbPI, SPCI, cross Venn-Abers, normalized and Mondrian split, against other libraries and the papers' own code |
| `exactness.py` | Split-conformal threshold against the exact value, several packages |
| `exact_rule_simulation.py` | Plain quantile against the exact finite-sample rule |

`speed_timeseries.py` takes two optional arguments: a folder holding clones of
`aangelopoulos/conformal-time-series` and `isgibbs/DtACI`, and the path to
`Rscript`. Sections whose reference is missing are skipped and say so.

Results depend on package versions and on the machine. The front-page figures
were measured on 20 and 21 November 2026 on 2 cores, with the versions named
there.

`reports/` holds short, self-contained scripts that reproduce the faults reported to other packages (TorchCP, crepes, PUNCC) and the MAPIE EnbPI timing. They need only the package concerned.

`review_experiments.py` holds the extra experiments of the technical paper: latency one row at a time with a stub model, stress cases for Venn-Abers, the approximate forest mode, very large label sets, and the CIR ablation.
