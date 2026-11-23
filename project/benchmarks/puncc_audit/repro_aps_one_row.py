"""PUNCC 0.9.3: APS asked for one prediction set at a time, as a deployed model is. Prints coverage at a 90% target."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from deel.puncc.classification import APS
from deel.puncc.api.prediction import IdPredictor

g = np.random.default_rng(0); K = 10
def draw(n):
    P = g.dirichlet(np.ones(K) * 0.3, n)                     # predicted class probabilities
    return P, np.array([g.choice(K, p=p) for p in P])        # labels drawn from them: a perfectly calibrated model
P_cal, y_cal = draw(1000); P_test, y_test = draw(1000)

aps = APS(IdPredictor(), train=False)
aps.fit(X_calib=P_cal, y_calib=y_cal)
batch = aps.predict(P_test, alpha=0.1)[-1]
one = [aps.predict(P_test[i:i + 1], alpha=0.1)[-1][0] for i in range(1000)]
for name, sets in (("all 1,000 rows in one call", batch), ("one row per call", one)):
    print(f"{name:27s}: coverage {np.mean([y_test[i] in sets[i] for i in range(1000)]):.3f}, mean set size {np.mean([len(s) for s in sets]):.2f}")
