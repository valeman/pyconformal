# Case studies on real series

Reproduces `project/docs/case_studies.md`.

Data: three files from the Monash Time Series Forecasting Archive (forecastingdata.org): `us_births_dataset.tsf`, `saugeenday_dataset.tsf`, `australian_electricity_demand_dataset.tsf`. Put them in one folder and point `PYCONFORMAL_DATA` at it.

```
pip install statsforecast plotly kaleido pandas
export PYCONFORMAL_DATA=/path/to/tsf/files
for s in births river power; do python run_sf.py $s snaive 1500; done
python run_sf.py river ets 1200
python run_sf.py births ets 200
python run_sf.py power theta 400
python analyse.py      # the tables
python figures.py      # the figures, in out/
```

`run_sf.py` calls statsforecast once per forecast origin, as a user would, and saves its point forecasts and its own 90% intervals. `analyse.py` gives the same point forecasts to pyconformal.
