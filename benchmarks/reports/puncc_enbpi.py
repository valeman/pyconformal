import numpy as np
from importlib.metadata import version
from sklearn.linear_model import LinearRegression
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.regression import EnbPI
print("puncc", version("puncc"))
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3)); y = X @ [1.0, 2.0, 3.0] + rng.normal(size=300)
m = EnbPI(BasePredictor(LinearRegression()), B=30, random_state=0)
m.fit(X[:200], y[:200])
print("s=2 ok:", m.predict(X[200:], alpha=0.1, y_true=y[200:], s=2)[0].shape)
m.predict(X[200:], alpha=0.1, y_true=y[200:], s=1)
