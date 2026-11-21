"""Rolling-origin forecasts with statsforecast's own conformal intervals.
usage: run_sf.py series model n_origins"""
import sys, time, numpy as np, warnings
warnings.filterwarnings("ignore")
from common import *
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive, AutoTheta
from statsforecast.utils import ConformalIntervals

series, model, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
if series == "births":
    y = read_tsf("us_births_dataset.tsf")["T1"][1]; freq, H, m, hist = "D", 7, 7, 364
elif series == "river":
    y = read_tsf("saugeenday_dataset.tsf")["T1"][1]; freq, H, m, hist = "D", 7, 1, 364
elif series == "power":
    v = read_tsf("australian_electricity_demand_dataset.tsf")["T2"][1]
    y = v[: len(v) // 2 * 2].reshape(-1, 2).mean(axis=1); freq, H, m, hist = "h", 24, 24, 24 * 28
mk = {"snaive": lambda: SeasonalNaive(season_length=m), "ets": lambda: AutoETS(season_length=m),
      "theta": lambda: AutoTheta(season_length=m)}[model]
T = len(y); first = T - H - N + 1          # origins first-1 .. : history ends at index o (exclusive)
mean = np.full((N, H), np.nan); lo = {2: mean.copy(), 10: mean.copy()}; hi = {2: mean.copy(), 10: mean.copy()}
t0 = time.time()
for i in range(N):
    o = first + i - 1 + 1                   # history is y[:o], targets y[o:o+H]
    df = sf_frame(y[o - hist:o], "2000-01-01", freq)
    for nw in (2, 10):
        sf = StatsForecast(models=[mk()], freq=freq, n_jobs=1)
        f = sf.forecast(df=df, h=H, level=[90], prediction_intervals=ConformalIntervals(h=H, n_windows=nw))
        c = f.columns[2]
        mean[i] = f[c].to_numpy(); lo[nw][i] = f[c + "-lo-90"].to_numpy(); hi[nw][i] = f[c + "-hi-90"].to_numpy()
    if i in (5, 50) or i % 200 == 0:
        print(series, model, i, round(time.time() - t0, 1), flush=True)
origins = first + np.arange(N)
np.savez(f"sf_{series}_{model}.npz", y=y, origins=origins, mean=mean, lo2=lo[2], hi2=hi[2], lo10=lo[10], hi10=hi[10], H=H)
print("done", series, model, round(time.time() - t0, 1))
