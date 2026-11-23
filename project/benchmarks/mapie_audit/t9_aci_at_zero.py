"""What MAPIE's ACI returns once the working level has been clipped to 0, with and without allow_infinite_bounds."""
import numpy as np, warnings
from sklearn.model_selection import PredefinedSplit
from common import *
from mapie.regression import TimeSeriesRegressor
warnings.simplefilter("ignore")
n0 = 200
y = np.arange(1.0, n0 + 40) ** 2            # every outcome larger than all before
X = np.zeros((len(y), 1))
m = TimeSeriesRegressor(Const(0.0), method="aci", cv=PredefinedSplit([-1] + [0] * (n0 - 1)), random_state=0).fit(X[:n0], y[:n0])
for t in range(n0, n0 + 12):
    out = []
    for inf in (True, False):
        try:
            _, iv = m.predict(X[t:t+1], confidence_level=0.9, ensemble=False, allow_infinite_bounds=inf)
            out.append(f"[{iv[0,0,0]:.0f}, {iv[0,1,0]:.0f}]")
        except Exception as e:
            out.append(f"{type(e).__name__}: {str(e)[:70]}")
    a = list(m.current_alpha.values())[0] if hasattr(m, "current_alpha") else None
    print(f"t={t} y={y[t]:.0f} level before={a}  infinite allowed: {out[0]}   default: {out[1]}")
    m.adapt_conformal_inference(X[t:t+1], y[t:t+1], gamma=0.05)
    m.update(X[t:t+1], y[t:t+1])
