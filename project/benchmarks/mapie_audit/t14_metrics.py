"""Metrics: each function against a hand calculation; calibration tests against their stated error rate on perfectly calibrated data."""
import numpy as np, warnings, sys
warnings.simplefilter("ignore")
from mapie.metrics.regression import regression_coverage_score, regression_mean_width_score, regression_mwi_score, regression_ssc, regression_ssc_score, hsic, coverage_width_based
from mapie.metrics.classification import classification_coverage_score, classification_mean_width_score, classification_ssc, classification_ssc_score
from mapie.metrics.calibration import expected_calibration_error, top_label_ece, kolmogorov_smirnov_p_value, kuiper_p_value, spiegelhalter_p_value, spiegelhalter_statistic, kolmogorov_smirnov_statistic, kuiper_statistic
rng = np.random.default_rng(0); ok = lambda a, b, tol=1e-9: "ok" if np.allclose(a, b, atol=tol, rtol=1e-9) else f"DIFFERS (MAPIE {np.round(a, 6)}, by hand {np.round(b, 6)})"
n = 500; y = rng.normal(size=n); lo = rng.normal(-1.3, 0.3, n); hi = lo + rng.gamma(4, 0.6, n); iv = np.stack([lo, hi], axis=1)[:, :, None]
print("regression coverage", ok(regression_coverage_score(y, iv)[0], ((lo <= y) & (y <= hi)).mean()))
print("regression mean width", ok(regression_mean_width_score(iv)[0], (hi - lo).mean()))
a = 0.1; win = ((hi - lo) + 2 / a * np.clip(lo - y, 0, None) + 2 / a * np.clip(y - hi, 0, None)).mean()
print("mean Winkler interval score", ok(regression_mwi_score(y, iv, 0.9), win))
# size-stratified coverage: worst coverage over width bins
try:
    ssc = regression_ssc(y, iv, num_bins=3)[0]; order = np.argsort(hi - lo); bins = np.array_split(order, 3)
    print("regression SSC (3 width bins)", ok(np.sort(ssc), np.sort([((lo[b] <= y[b]) & (y[b] <= hi[b])).mean() for b in bins])), "; SSC score", ok(regression_ssc_score(y, iv, num_bins=3)[0], min(((lo[b] <= y[b]) & (y[b] <= hi[b])).mean() for b in bins)))
except Exception as e: print("regression SSC ERROR", type(e).__name__, str(e)[:120])
# classification
K = 5; yc = rng.integers(0, K, n); S = rng.random((n, K)) < 0.5; S3 = S[:, :, None]
print("classification coverage", ok(classification_coverage_score(yc, S3)[0], S[np.arange(n), yc].mean()))
print("classification mean set size", ok(classification_mean_width_score(S3)[0], S.sum(1).mean()))
# ECE with equal-width bins, binary
p = rng.random(4000); yb = (rng.random(4000) < p ** 1.5).astype(int)
def ece_hand(yb, p, B):
    e = 0.0; edges = np.linspace(0, 1, B + 1); idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, B - 1)
    for b in range(B):
        m = idx == b
        if m.any(): e += m.sum() * abs(yb[m].mean() - p[m].mean())
    return e / len(p)
for B in (10, 50):
    for strat in (None, "uniform"):
        try: print(f"ECE, {B} bins, split_strategy={strat}", ok(expected_calibration_error(yb, p, num_bins=B, split_strategy=strat), ece_hand(yb, p, B), 1e-6))
        except Exception as e: print(f"ECE {B} {strat} ERROR", type(e).__name__, str(e)[:100])
# Spiegelhalter statistic by its formula
z = np.sum((yb - p) * (1 - 2 * p)) / np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
print("Spiegelhalter statistic", ok(spiegelhalter_statistic(yb, p), z))
# Calibration tests: share of p-values below 0.05 when the scores are perfectly calibrated (should be about 0.05), and when they are not
R = int(sys.argv[1])
for name, f in (("Kolmogorov-Smirnov", kolmogorov_smirnov_p_value), ("Kuiper", kuiper_p_value), ("Spiegelhalter", spiegelhalter_p_value)):
    for nn in (200, 2000):
        r0 = r1 = 0; pv = []
        for r in range(R):
            g = np.random.default_rng(1299709 * r + nn); s = g.random(nn)
            v = f((g.random(nn) < s).astype(int), s); pv.append(v); r0 += v < 0.05
            r1 += f((g.random(nn) < s ** 1.3).astype(int), s) < 0.05
        se = np.sqrt(0.05 * 0.95 / R)
        print(f"{name:18s} n={nn:4d}: rejects at 5% on calibrated scores in {r0/R:.3f} of runs {'ABOVE 5%' if r0/R > 0.05 + 3*se else ('far below 5%' if r0/R < 0.05 - 3*se else 'ok')}; on miscalibrated scores {r1/R:.3f}; p-value range [{min(pv):.3f}, {max(pv):.3f}], mean {np.mean(pv):.3f}", flush=True)
