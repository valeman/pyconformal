"""Multi-step split conformal prediction: pyconformal against the R package
conformalForecast (function scp). Usage: python speed_multistep.py /path/to/Rscript

3,000 time steps, 12 horizons, 90% intervals, expanding calibration.
"""
import os, subprocess, sys, tempfile, time
import numpy as np
import pyconformal as pc

rng = np.random.default_rng(0); T, H = 3000, 12
y = np.zeros(T)
for t in range(1, T): y[t] = 0.8 * y[t - 1] + rng.normal()
F = np.full((T, H), np.nan)
for h in range(1, H + 1): F[h:, h - 1] = 0.8**h * y[:-h]
ms = pc.MultiStepSplitConformal(alpha=0.1, min_calibration=50); ms.run(y_pred=F[:200], y=y[:200])
best = 1e9
for _ in range(5):
    t = time.perf_counter(); r = ms.run(y_pred=F, y=y); best = min(best, time.perf_counter() - t)
print(f"pyconformal: {best * 1e3:.1f} ms; coverage by horizon {np.round(r.coverage, 3)}")
if len(sys.argv) > 1:
    d = tempfile.mkdtemp(); np.savetxt(os.path.join(d, "y.txt"), y); np.savetxt(os.path.join(d, "F.csv"), F, delimiter=",")
    code = f'''suppressMessages(library(conformalForecast))
y <- scan("{d}/y.txt", quiet=TRUE); Fm <- as.matrix(read.csv("{d}/F.csv", header=FALSE)); Fm[is.nan(Fm)] <- NA; h <- ncol(Fm)
obj <- structure(list(x=ts(y), series="y", MEAN=ts(rbind(Fm, matrix(0, h, h))), ERROR=ts(y - Fm), level=c(90), forward=TRUE), class=c("cvforecast","forecast"))
t0 <- proc.time()[3]; s <- scp(obj, alpha=0.1, symmetric=FALSE, ncal=50, rolling=FALSE); cat(proc.time()[3]-t0, "\\n")
write.csv(as.matrix(s$LOWER[[1]]), "{d}/lo.csv", row.names=FALSE); write.csv(as.matrix(s$UPPER[[1]]), "{d}/up.csv", row.names=FALSE)'''
    sec = float(subprocess.run([sys.argv[1], "-e", code], capture_output=True, text=True).stdout.split()[-1])
    lo = np.genfromtxt(os.path.join(d, "lo.csv"), delimiter=",", skip_header=1)[:T]; up = np.genfromtxt(os.path.join(d, "up.csv"), delimiter=",", skip_header=1)[:T]
    both = ~np.isnan(lo) & ~np.isnan(r.intervals[:, :, 0])
    diff = np.maximum(np.abs(lo - r.intervals[:, :, 0]), np.abs(up - r.intervals[:, :, 1]))[both]
    print(f"R conformalForecast scp: {sec:.1f} s ({sec / best:,.0f} times slower); {both.sum():,} intervals compared, {(diff > 1e-9).sum()} differ")

# AcMCP: 1,000 time steps, 6 horizons, scorecaster on (an MA model and a
# regression are refitted at every step for every horizon above 1).
T, H = 1000, 6
rng = np.random.default_rng(2); e = rng.standard_t(5, T); y = np.zeros(T)
for t in range(1, T): y[t] = 0.8 * y[t - 1] + e[t] + 0.5 * e[t - 1]
F = np.column_stack([np.r_[np.full(h - 1, np.nan), 0.7**h * y[: T - h]] for h in range(1, H + 1)]); y = y[1:]
ac = pc.AcMCP(alpha=0.1, min_calibration=50, KI=2.0, Csat=1.5); ac.run(y_pred=F[:120], y=y[:120])
t = time.perf_counter(); r = ac.run(y_pred=F, y=y); ours = time.perf_counter() - t
t = time.perf_counter(); pc.AcMCP(alpha=0.1, min_calibration=50, KI=2.0, Csat=1.5, scorecast=False).run(y_pred=F, y=y); plain = time.perf_counter() - t
print(f"AcMCP pyconformal: {ours:.2f} s with the scorecaster, {plain * 1e3:.1f} ms without")
if len(sys.argv) > 1:
    d = tempfile.mkdtemp(); np.savetxt(os.path.join(d, "y.txt"), y); np.savetxt(os.path.join(d, "F.csv"), F, delimiter=",")
    code = f'''suppressMessages(library(conformalForecast))
y <- scan("{d}/y.txt", quiet=TRUE); Fm <- as.matrix(read.csv("{d}/F.csv", header=FALSE)); Fm[is.nan(Fm)] <- NA; h <- ncol(Fm)
obj <- structure(list(x=ts(y), series="y", MEAN=ts(rbind(Fm, matrix(0, h, h))), ERROR=ts(y - Fm), level=c(90), forward=TRUE), class=c("cvforecast","forecast"))
for (sc in c(FALSE, TRUE)) {{ t0 <- proc.time()[3]; s <- acmcp(obj, alpha=0.1, ncal=50, scorecast=sc, KI=2.0, Csat=1.5); cat(proc.time()[3]-t0, "\\n") }}
write.csv(as.matrix(s$LOWER[[1]]), "{d}/lo.csv", row.names=FALSE); write.csv(as.matrix(s$UPPER[[1]]), "{d}/up.csv", row.names=FALSE)'''
    secs = [float(v) for v in subprocess.run([sys.argv[1], "-e", code], capture_output=True, text=True).stdout.split()[-2:]]
    n = len(y); lo = np.genfromtxt(os.path.join(d, "lo.csv"), delimiter=",", skip_header=1)[:n]; up = np.genfromtxt(os.path.join(d, "up.csv"), delimiter=",", skip_header=1)[:n]
    diff = np.maximum(np.abs(lo - r.intervals[:, :, 0]), np.abs(up - r.intervals[:, :, 1])); diff = diff[~np.isnan(diff)]
    print(f"R acmcp: {secs[1]:.0f} s with the scorecaster ({secs[1] / ours:.0f} times slower), {secs[0]:.1f} s without ({secs[0] / plain:,.0f} times slower)")
    print(f"difference from R with the scorecaster: median {np.median(diff):.1e}, largest {diff.max():.1e}")
