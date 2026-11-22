"""Split conformal classification: every score and every include_last_label option.
Probabilities are exact (labels are drawn from them), so the data are exchangeable and the model is perfect."""
import sys, numpy as np
from common import *
from mapie.classification import SplitConformalClassifier
R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
K, NT = 5, 1000
print("score      last_label   n_cal level  coverage (se)      mean size  verdict")
for score in ("lac", "top_k", "aps", "raps"):
    opts = (None,) if score in ("lac", "top_k") else (True, False, "randomized")
    for opt in opts:
        for n_cal in (50, 200, 1000):
            for level in (0.9, 0.8):
                cov, size, err = [], [], None
                for r in range(R):
                    rng = np.random.default_rng(100000 * r + n_cal)
                    P = rng.dirichlet(np.ones(K) * 0.7, n_cal + NT)
                    y = (P.cumsum(1) > rng.random((len(P), 1))).argmax(1)
                    try:
                        m = SplitConformalClassifier(FixedClf(K).fit(P), confidence_level=level, conformity_score=score, prefit=True, random_state=r)
                        m.conformalize(P[:n_cal], y[:n_cal])
                        kw = {} if opt is None else {"conformity_score_params": {"include_last_label": opt}}
                        S = m.predict_set(P[n_cal:], **kw)[1][:, :, 0]
                        cov.append(S[np.arange(NT), y[n_cal:]].mean()); size.append(S.sum(1).mean())
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:60]}"; break
                if err: print(f"{score:10s} {str(opt):12s} {n_cal:5d} {level}   ERROR {err}", flush=True); continue
                m_, se = summary(cov)
                print(f"{score:10s} {str(opt):12s} {n_cal:5d} {level}   {m_:.4f} ({se:.4f})   {np.mean(size):.2f}       {verdict(m_, se, level)}", flush=True)
