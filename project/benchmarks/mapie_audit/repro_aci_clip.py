import numpy as np, warnings
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import PredefinedSplit
from mapie.regression import TimeSeriesRegressor
warnings.simplefilter("ignore")

n0, T, gamma, alpha = 200, 3000, 0.05, 0.1
rng = np.random.default_rng(0)
y = rng.normal(size=n0 + T) * 1.01 ** np.arange(n0 + T)   # noise that grows 1% per step
X = np.zeros((n0 + T, 1))

m = TimeSeriesRegressor(DummyRegressor(strategy="constant", constant=0.0), method="aci",
                        cv=PredefinedSplit([-1] + [0] * (n0 - 1)), random_state=0).fit(X[:n0], y[:n0])
miss = []
for t in range(n0, n0 + T):
    _, iv = m.predict(X[t:t+1], confidence_level=1 - alpha, ensemble=False, allow_infinite_bounds=True)
    miss.append(not (iv[0, 0, 0] <= y[t] <= iv[0, 1, 0]))
    m.adapt_conformal_inference(X[t:t+1], y[t:t+1], gamma=gamma)
    m.update(X[t:t+1], y[t:t+1])

bound = (max(alpha, 1 - alpha) + gamma) / (gamma * T)
print(f"miss rate {np.mean(miss):.4f}; Gibbs & Candes Prop. 4.1 allows at most {alpha + bound:.4f}")
