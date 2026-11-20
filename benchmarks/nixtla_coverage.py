"""Coverage of statsforecast and mlforecast conformal intervals on the real libraries.
Each series is i.i.d. N(0,1) noise, so the errors of a mean forecast are exchangeable and a valid
90% conformal interval must cover at least 90% of the time."""
import warnings, time; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsforecast, mlforecast
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage
from statsforecast.utils import ConformalIntervals
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from sklearn.linear_model import LinearRegression

N, T, LEVEL = 4000, 120, 90
rng = np.random.default_rng(0)
y = rng.standard_normal((N, T + 1))
ds = pd.date_range("2000-01-01", periods=T + 1, freq="D")
df = pd.DataFrame({"unique_id": np.repeat(np.arange(N), T), "ds": np.tile(ds[:T], N), "y": y[:, :T].ravel()})
truth = pd.Series(y[:, T], index=np.arange(N))
print(f"statsforecast {statsforecast.__version__}, mlforecast {mlforecast.__version__}; {N} series of length {T}, nominal {LEVEL}%")

def cover(fc, lo, hi):
    fc = fc.set_index("unique_id").loc[np.arange(N)]
    t = truth.values
    return np.mean((fc[lo].values <= t) & (t <= fc[hi].values)), np.mean(fc[hi].values - fc[lo].values)

print("\nstatsforecast, HistoricAverage + ConformalIntervals(h=1, n_windows=w)")
for method in ["conformal_distribution", "conformal_error"]:
    for w in [2, 5, 10, 20]:
        sf = StatsForecast(models=[HistoricAverage(prediction_intervals=ConformalIntervals(h=1, n_windows=w, method=method))], freq="D", n_jobs=1)
        fc = sf.forecast(df=df, h=1, level=[LEVEL])
        c, wd = cover(fc, f"HistoricAverage-lo-{LEVEL}", f"HistoricAverage-hi-{LEVEL}")
        print(f"  {method:23} n_windows={w:3}: coverage {c:.3f}  mean width {wd:.2f}")

print("\nmlforecast, LinearRegression on lag 1 + PredictionIntervals(n_windows=w, h=1)")
for method in ["conformal_distribution", "conformal_error"]:
    for w in [2, 5, 10, 20]:
        mlf = MLForecast(models={"lr": LinearRegression()}, freq="D", lags=[1])
        mlf.fit(df, prediction_intervals=PredictionIntervals(n_windows=w, h=1, method=method))
        fc = mlf.predict(1, level=[LEVEL])
        c, wd = cover(fc, f"lr-lo-{LEVEL}", f"lr-hi-{LEVEL}")
        print(f"  {method:23} n_windows={w:3}: coverage {c:.3f}  mean width {wd:.2f}")
import inspect
print("\ndefaults:", "ConformalIntervals", inspect.signature(ConformalIntervals.__init__), "| PredictionIntervals", inspect.signature(PredictionIntervals.__init__))
print("normal-theory 90% width for N(0,1):", round(2 * 1.6449, 2))
