# Benchmarks

Scripts behind every number on the front page. Run them in a separate
environment so the audited packages do not mix with your own:

```bash
conda create -n cp-audit python=3.12 -y && conda activate cp-audit
pip install pyconformal[fast] statsforecast mlforecast darts mapie venn-abers crepes puncc scikit-learn
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
| `exactness.py` | Split-conformal threshold against the exact value, several packages |
| `exact_rule_simulation.py` | Plain quantile against the exact finite-sample rule |

Results depend on package versions. The front-page figures were measured on
20 November 2026 with the versions named there.
