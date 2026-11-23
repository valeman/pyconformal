"""PUNCC 0.9.3: APS and RAPS, the same fitted object asked for sets in batches of different sizes.
prediction_sets.raps_set caps the set size at the number of TEST ROWS in the call (line 192) where the number of classes is meant."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from deel.puncc.classification import APS, RAPS, LAC
from deel.puncc.api.prediction import IdPredictor
K = 10
def draw(g, n):
    P = g.dirichlet(np.ones(K) * 0.3, n); y = np.array([g.choice(K, p=p) for p in P]); return P, y
R = 60
for name, make in (("APS, rand=True", lambda: APS(IdPredictor(), train=False)), ("APS, rand=False", lambda: APS(IdPredictor(), train=False, rand=False)),
                   ("RAPS, lambd=0.01, k_reg=2", lambda: RAPS(IdPredictor(), train=False, lambd=0.01, k_reg=2)), ("LAC", lambda: LAC(IdPredictor(), train=False))):
    for batch in (1, 2, 5, 20, 1000):
        cov = []; size = []; err = None
        for r in range(R):
            g = np.random.default_rng(r); Pc, yc = draw(g, 1000); Pt, yt = draw(g, 1000)
            try:
                m = make(); m.fit(X_calib=Pc, y_calib=yc); sets = []
                for i in range(0, 1000 if batch >= 20 else 200, batch):
                    out = m.predict(Pt[i:i + batch], alpha=0.1); sets.extend(out[-1])
            except Exception as e: err = f"{type(e).__name__}: {' '.join(str(e).split())[:120]}"; break
            cov.append(np.mean([yt[i] in sets[i] for i in range(len(sets))])); size.append(np.mean([len(s) for s in sets]))
        if err: print(f"{name:26s} batch of {batch:4d}: ERROR {err}"); continue
        print(f"{name:26s} predicting {batch:4d} row(s) per call: coverage {np.mean(cov):.4f} (target 0.90), mean set size {np.mean(size):.2f}, largest set {max(size):.1f}", flush=True)
