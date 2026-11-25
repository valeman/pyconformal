"""conformal-tights 0.5.0 DartsForecaster: out-of-sample coverage of its quantile forecasts.
Part A: synthetic hourly series, AR(1) errors with coefficient 0.9, one-step forecasts, 5 seeds.
Part B: the README's example (Zurich household electricity, hourly), train before 1 June 2022, test after.
        B1 one-step forecasts over the test period; B2 the README's 5-day path forecast from 18 successive origins.
Defaults throughout (XGBoost defaults); quantiles through num_samples=500 as in the README."""
import warnings, time, logging, numpy as np, pandas as pd
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from darts import TimeSeries
from xgboost import XGBRegressor
from conformal_tights import ConformalCoherentQuantileRegressor as CT, DartsForecaster
Q = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
def cov_table(q, y):
    return {0.95: np.mean((q[:, 0] <= y) & (y <= q[:, -1])), 0.90: np.mean((q[:, 1] <= y) & (y <= q[:, -2])),
            0.80: np.mean((q[:, 2] <= y) & (y <= q[:, -3])), 0.50: np.mean((q[:, 3] <= y) & (y <= q[:, -4]))}
def one_step(f, ts, start, **kw):
    hf = f.historical_forecasts(ts, start=start, forecast_horizon=1, retrain=False, num_samples=500, last_points_only=True, predict_kwargs={"quantiles": Q}, verbose=False, **kw)
    q = np.quantile(hf.all_values()[:, 0, :], Q, axis=1).T
    y = ts.slice_intersect(hf).values()[:, 0]
    return q, y
print("Part A: synthetic AR(1) errors, 3,000 training hours, 1,000 one-step test forecasts")
rows = []
for seed in range(5):
    rng = np.random.default_rng(seed); T = 4000; e = np.zeros(T)
    for t in range(1, T): e[t] = 0.9 * e[t - 1] + rng.normal() * (1 + 0.5 * np.sin(2 * np.pi * t / 24))
    ts = TimeSeries.from_times_and_values(pd.date_range("2020-01-01", periods=T, freq="h"), 10 + 3 * np.sin(2 * np.pi * np.arange(T) / 24) + e)
    f = DartsForecaster(model=CT(random_state=seed), lags=24); f.fit(ts[:3000])
    q, y = one_step(f, ts, 3000); c = cov_table(q, y); rows.append([c[k] for k in (0.95, 0.9, 0.8, 0.5)]); print(f"  seed {seed}:", {k: round(v, 3) for k, v in c.items()}, flush=True)
print("  mean over seeds: 95%% %.3f  90%% %.3f  80%% %.3f  50%% %.3f" % tuple(np.mean(rows, 0)))
print("Part B: Zurich household electricity (README example)")
from darts.datasets import ElectricityConsumptionZurichDataset
ts = ElectricityConsumptionZurichDataset().load().resample("h")
y = ts["Value_NE5"]; X = ts.drop_columns(["Value_NE5", "Value_NE7"])
X = X.add_holidays(country_code="CH").add_datetime_attribute("month").add_datetime_attribute("dayofweek").add_datetime_attribute("hour")
cats = ["holidays", "month", "dayofweek", "hour"]
cut = pd.Timestamp("2022-06-01"); y_tr, y_te = y.split_after(cut); X_tr, X_te = X.split_after(cut)
f = DartsForecaster(model=CT(estimator=XGBRegressor()), lags=5 * 24, lags_future_covariates=[0], categorical_future_covariates=cats)
t0 = time.time(); f.fit(y_tr, future_covariates=X_tr); print(f"  fit on {len(y_tr)} hours: {time.time()-t0:.0f} s; level-2 calibration rows: {len(f.model.ŷ_calib_l2_)}", flush=True)
t0 = time.time(); q, yy = one_step(f, y, cut + pd.Timedelta(hours=1), future_covariates=X)
print(f"  B1 one-step, {len(yy)} test hours ({time.time()-t0:.0f} s):", {k: round(v, 3) for k, v in cov_table(q, yy).items()}, flush=True)
H = 120; qs, ys = [], []
origins = [cut + pd.Timedelta(hours=H * k) for k in range(18)]
origins = [o for o in origins if o + pd.Timedelta(hours=H) <= y.end_time()]
t0 = time.time()
for o in origins:
    fc = f.predict(n=H, series=y.drop_after(o + pd.Timedelta(hours=1)), future_covariates=X, num_samples=500, quantiles=Q)
    qs.append(np.quantile(fc.all_values()[:, 0, :], Q, axis=1).T); ys.append(y.slice_intersect(fc).values()[:, 0])
qs, ys = np.array(qs), np.array(ys)
print(f"  B2 README 5-day path forecast, {len(origins)} origins ({time.time()-t0:.0f} s)")
for name, sl in (("hours 1-24", slice(0, 24)), ("hours 25-72", slice(24, 72)), ("hours 73-120", slice(72, 120)), ("all 120", slice(0, 120))):
    print(f"    {name:13s}", {k: round(v, 3) for k, v in cov_table(qs[:, sl].reshape(-1, len(Q)), ys[:, sl].ravel()).items()})
print("Part C: historical_forecasts(predict_likelihood_parameters=True) with quantiles chosen at prediction time")
try:
    f.historical_forecasts(y, start=cut + pd.Timedelta(hours=1), forecast_horizon=1, retrain=False, predict_likelihood_parameters=True, last_points_only=True, predict_kwargs={"quantiles": (0.05, 0.5, 0.95)}, future_covariates=X, verbose=False)
    print("  ran")
except Exception as ex:
    print("  raises", type(ex).__name__ + ":", str(ex)[:160])
