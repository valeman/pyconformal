"""Coverage of Darts ConformalNaiveModel when the calibration set is short (i.i.d. N(0,1) series)."""
import warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
import numpy as np, darts
from darts import TimeSeries
from darts.models import ConformalNaiveModel, LinearRegressionModel
N, T = 1000, 40
rng = np.random.default_rng(0)
y = rng.standard_normal((N, T + 1))
series = [TimeSeries.from_values(v[:T].astype(np.float64)) for v in y]
base = LinearRegressionModel(lags=1).fit(series)
print(f"darts {darts.__version__}: {N} series, base LinearRegressionModel(lags=1), nominal 90% (quantiles 0.05, 0.95)")
for cal in [3, 5, 8, 9, 20, None]:
    m = ConformalNaiveModel(base, quantiles=[0.05, 0.5, 0.95], cal_length=cal)
    pr = m.predict(n=1, series=series, predict_likelihood_parameters=True)
    lo = np.array([p.values()[0, 0] for p in pr]); hi = np.array([p.values()[0, 2] for p in pr])
    t = y[:, T]
    print(f"  cal_length={str(cal):5}: coverage {np.mean((lo <= t) & (t <= hi)):.3f}  infinite {np.mean(~np.isfinite(hi)):.3f}  mean width {np.mean(hi - lo):.2f}")
