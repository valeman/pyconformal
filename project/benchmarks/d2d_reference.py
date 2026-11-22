"""ConformalDistributionRegressor against the code of the paper's authors
(github.com/tuanhacs/conformal_d2d), on their own mixture-of-betas data.

    git clone https://github.com/tuanhacs/conformal_d2d
    python d2d_reference.py path/to/conformal_d2d

Checks that coefficients, scores and radii are the same numbers, measures
coverage against the theory, and times both.
"""
import math
import sys
import time
import warnings

import numpy as np
import pyconformal as pc

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from utils.helpers import L2_distance, fourier_coeffs, load_data  # noqa: E402

R, ETA = 10, 1000
N = 3200
X_list, Y_list, _ = load_data("mixture_of_betas", N, ETA)
X, Y = np.asarray(X_list), np.asarray(Y_list)


def best(f, n=5):
    out = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); out.append(time.perf_counter() - t)
    return min(out), r


# 1. coefficients
t_ref, A_ref = best(lambda: np.array([fourier_coeffs(x, R) for x in X_list] + [fourier_coeffs(y, R) for y in Y_list]), 3)
pc.cosine_coefficients(X[:2], R)  # compile
t_new, A_new = best(lambda: np.vstack([pc.cosine_coefficients(X, R), pc.cosine_coefficients(Y, R)]))
print(f"coefficients, {2 * N} distributions x {ETA} points x {R} terms: authors {t_ref:.3f} s, pyconformal {t_new:.4f} s "
      f"({t_ref / t_new:.0f}x); max abs difference {np.abs(A_ref[:, 1:] - A_new).max():.1e}")
from pyconformal import distribution as D  # noqa: E402
out = np.empty((N, R)); t_py, _ = best(lambda: D._coef_py(X, out), 3)
print(f"  without numba: {2 * t_py:.3f} s ({t_ref / (2 * t_py):.1f}x)")
A_in, A_out = A_new[:N], A_new[N:]

# 2. the same model for both: ridge from input to output coefficients
tr, cal, te = slice(0, 1000), slice(1000, 2000), slice(2000, N)
model = Ridge(1e-4).fit(A_in[tr], A_out[tr])
P = model.predict(A_in)
alphas = np.linspace(0.04, 0.30, 100)

def ref_global():
    s = [L2_distance(A_out[i], P[i]) for i in range(1000, 2000)]
    return np.array([np.sort(s)[int(np.ceil((1000 + 1) * (1 - a))) - 1] for a in alphas])

def new_global():
    m = pc.ConformalDistributionRegressor().calibrate(y_pred=P[cal], y=A_out[cal])
    return np.array([m.radius(alpha=float(a)) for a in alphas])

t_ref, q_ref = best(ref_global, 3); t_new, q_new = best(new_global)
print(f"global radius at 100 levels: authors {t_ref * 1e3:.1f} ms, pyconformal {t_new * 1e3:.1f} ms ({t_ref / t_new:.0f}x); "
      f"levels where the radius differs: {int((~np.isclose(q_ref, q_new, rtol=1e-12)).sum())} of 100")

# 3. localised radius, K = 500, 1,200 test cases, 10 levels (the authors' loop is per case and per level)
K, some = 500, alphas[::10]
def ref_local():
    nn = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(A_in[cal])
    s = np.array([L2_distance(A_out[i], P[i]) for i in range(1000, 2000)])
    out = np.empty((len(some), N - 2000))
    for i, a in enumerate(some):
        for j in range(2000, N):
            idx = nn.kneighbors(A_in[j].reshape(1, -1), n_neighbors=K, return_distance=True)[1][0]
            k_star = max(0, min(int(math.ceil((K + 1) * (1 - a))) - 1, K - 1))
            out[i, j - 2000] = np.sort(s[idx])[k_star]
    return out

def new_local():
    m = pc.ConformalDistributionRegressor(k_neighbors=K).calibrate(y_pred=P[cal], y=A_out[cal], features=A_in[cal])
    return m.radius(alpha=[float(a) for a in some], features=A_in[te]).T

t_ref, l_ref = best(ref_local, 1); t_new, l_new = best(new_local, 3)
print(f"localised radius, K={K}, {N - 2000} cases, {len(some)} levels: authors {t_ref:.2f} s, pyconformal {t_new:.3f} s "
      f"({t_ref / t_new:.0f}x); share of radii equal: {np.isclose(l_ref, l_new, rtol=1e-9).mean():.4f}")

# 4. coverage over fresh draws against the theory (mean coverage = ceil((n+1)(1-a)) / (n+1))
cov, loc = [], []
for seed in range(20):
    Xs, Ys, _ = load_data("mixture_of_betas", 1500, 300, seed=1000 + seed)
    a_in, a_out = pc.cosine_coefficients(np.asarray(Xs), R), pc.cosine_coefficients(np.asarray(Ys), R)
    mdl = Ridge(1e-4).fit(a_in[:500], a_out[:500])
    cd = pc.ConformalDistributionRegressor(mdl, alpha=0.1).calibrate(a_in[500:1000], a_out[500:1000])
    cov.append(cd.contains(a_out[1000:], a_in[1000:]).mean())
    cl = pc.ConformalDistributionRegressor(mdl, alpha=0.1, k_neighbors=200).calibrate(a_in[500:1000], a_out[500:1000])
    loc.append(cl.contains(a_out[1000:], a_in[1000:]).mean())
print(f"coverage at 90%, 20 fresh datasets, 500 calibration and 500 test pairs: global {np.mean(cov):.4f} "
      f"(theory {math.ceil(501 * 0.9) / 501:.4f}), localised K=200 {np.mean(loc):.4f}")
