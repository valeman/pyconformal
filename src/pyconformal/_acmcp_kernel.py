"""Kernels for AcMCP (autocorrelated multi-step conformal prediction).

Three pieces, all loops over time:

* the mean of an MA(q) model fitted by exact Gaussian maximum likelihood on a
  growing (or sliding) set of errors. The likelihood uses the innovations
  algorithm, which settles to O(q) per observation once its coefficients have
  converged, and each fit starts from the previous one;
* the scorecaster: for h = 1 the mean error, for h > 1 the average of the
  MA(h - 1) mean and a linear regression of the h-step error on the shorter
  horizons from the same forecast origin;
* quantile tracking with an integrator, one pass per horizon and side.

With numba the loops are compiled and the horizons of the MA fits run in
parallel; without it the same functions run as plain Python.
"""

from __future__ import annotations

import math

import numpy as np

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        def wrap(f):
            return f

        return wrap


@njit(cache=True, nogil=True)
def _ma_obj(theta, x, lo, n, q, TH, V, IY, I1, g, thf):
    """Concentrated -2 log-likelihood / n of an MA(q) with a mean, and the
    mean that maximises it (generalised least squares for these theta)."""
    R = q + 1
    for k in range(q + 1):
        s = 0.0
        for j in range(q - k + 1):
            a = 1.0 if j == 0 else theta[j - 1]
            b = 1.0 if j + k == 0 else theta[j + k - 1]
            s += a * b
        g[k] = s
    frozen = False
    vf = g[0]
    s11 = 0.0
    s1y = 0.0
    syy = 0.0
    slog = 0.0
    for i in range(n):
        row = i % R
        if not frozen:
            for j in range(R):
                TH[row, j] = 0.0
            kmin = i - q if i > q else 0
            for k in range(kmin, i):
                ssum = g[i - k]
                kr = k % R
                for j in range(kmin, k):
                    ssum -= TH[kr, k - j] * TH[row, i - j] * V[j % R]
                TH[row, i - k] = ssum / V[kr]
            v = g[0]
            for j in range(kmin, i):
                c = TH[row, i - j]
                v -= c * c * V[j % R]
            if not v > 1e-300:
                return math.inf, 0.0
            V[row] = v
            if i > q:
                prev = (i - 1) % R
                d = abs(v - V[prev])
                for j in range(1, R):
                    dd = abs(TH[row, j] - TH[prev, j])
                    if dd > d:
                        d = dd
                if d < 1e-13:
                    frozen = True
                    vf = v
                    for j in range(R):
                        thf[j] = TH[row, j]
            py = 0.0
            p1 = 0.0
            top = i if i < q else q
            for j in range(1, top + 1):
                c = TH[row, j]
                py += c * IY[(i - j) % R]
                p1 += c * I1[(i - j) % R]
        else:
            v = vf
            py = 0.0
            p1 = 0.0
            for j in range(1, R):
                c = thf[j]
                py += c * IY[(i - j) % R]
                p1 += c * I1[(i - j) % R]
        iy = x[lo + i] - py
        i1 = 1.0 - p1
        IY[row] = iy
        I1[row] = i1
        s11 += i1 * i1 / v
        s1y += i1 * iy / v
        syy += iy * iy / v
        slog += math.log(v)
    mu = s1y / s11
    rss = syy - s1y * s1y / s11
    if not rss > 0.0:
        return math.inf, mu
    return math.log(rss / n) + slog / n, mu


@njit(cache=True, nogil=True)
def _ma_grad(theta, x, lo, n, q, TH, V, IY, I1, g, thf, grad):
    for j in range(q):
        keep = theta[j]
        step = 1e-5 * (1.0 if abs(keep) < 1.0 else abs(keep))
        theta[j] = keep + step
        fp = _ma_obj(theta, x, lo, n, q, TH, V, IY, I1, g, thf)[0]
        theta[j] = keep - step
        fm = _ma_obj(theta, x, lo, n, q, TH, V, IY, I1, g, thf)[0]
        theta[j] = keep
        grad[j] = (fp - fm) / (2.0 * step)


@njit(cache=True, nogil=True)
def _ma_fit(x, lo, n, q, theta, Hinv, probe):
    """Maximum-likelihood mean of an MA(q) on x[lo:lo+n]; ``theta`` is the
    starting point and ``Hinv`` the starting inverse Hessian; both are
    overwritten, so the next fit on nearly the same data needs one or two
    steps. Returns (mean, objective); the mean is NaN on failure."""
    R = q + 1
    TH = np.zeros((R, R))
    V = np.zeros(R)
    IY = np.zeros(R)
    I1 = np.zeros(R)
    g = np.zeros(R)
    thf = np.zeros(R)
    grad = np.zeros(q)
    gnew = np.zeros(q)
    d = np.zeros(q)
    trial = np.zeros(q)
    f, mu = _ma_obj(theta, x, lo, n, q, TH, V, IY, I1, g, thf)
    if not math.isfinite(f):
        for j in range(q):
            theta[j] = 0.0
        f, mu = _ma_obj(theta, x, lo, n, q, TH, V, IY, I1, g, thf)
        if not math.isfinite(f):
            return math.nan, math.inf
    _ma_grad(theta, x, lo, n, q, TH, V, IY, I1, g, thf, grad)
    escaped = False
    for _ in range(300):
        gmax = 0.0
        for j in range(q):
            if abs(grad[j]) > gmax:
                gmax = abs(grad[j])
        if gmax < 1e-7:
            # theta and its mirror image across the unit circle have the same
            # likelihood, so the boundary between them is a stationary point
            # that is not a maximum. Step off it once and carry on.
            if escaped or not probe:
                break
            escaped = True
            moved = False
            for j in range(q):
                for sgn in (-1.0, 1.0):
                    for jj in range(q):
                        trial[jj] = theta[jj]
                    trial[j] += sgn * 0.01
                    ft, mt = _ma_obj(trial, x, lo, n, q, TH, V, IY, I1, g, thf)
                    if ft < f - 1e-10:
                        f = ft
                        mu = mt
                        for jj in range(q):
                            theta[jj] = trial[jj]
                        moved = True
            if not moved:
                break
            for i in range(q):
                for j in range(q):
                    Hinv[i, j] = 1.0 if i == j else 0.0
            _ma_grad(theta, x, lo, n, q, TH, V, IY, I1, g, thf, grad)
            continue
        slope = 0.0
        dmax = 0.0
        for i in range(q):
            s = 0.0
            for j in range(q):
                s -= Hinv[i, j] * grad[j]
            d[i] = s
            slope += s * grad[i]
            if abs(s) > dmax:
                dmax = abs(s)
        if not slope < 0.0:
            for i in range(q):
                for j in range(q):
                    Hinv[i, j] = 1.0 if i == j else 0.0
                d[i] = -grad[i]
            slope = 0.0
            dmax = 0.0
            for i in range(q):
                slope += d[i] * grad[i]
                if abs(d[i]) > dmax:
                    dmax = abs(d[i])
        step = 1.0 if dmax <= 0.37 else 0.37 / dmax
        done = False
        fnew = f
        munew = mu
        for _ls in range(40):
            for j in range(q):
                trial[j] = theta[j] + step * d[j]
            fnew, munew = _ma_obj(trial, x, lo, n, q, TH, V, IY, I1, g, thf)
            if fnew <= f + 1e-4 * step * slope:
                done = True
                break
            step *= 0.5
        if not done:
            break
        _ma_grad(trial, x, lo, n, q, TH, V, IY, I1, g, thf, gnew)
        sy = 0.0
        for j in range(q):
            sy += (trial[j] - theta[j]) * (gnew[j] - grad[j])
        if sy > 1e-14:
            # BFGS update of the inverse Hessian
            Hy = np.zeros(q)
            yHy = 0.0
            for i in range(q):
                s = 0.0
                for j in range(q):
                    s += Hinv[i, j] * (gnew[j] - grad[j])
                Hy[i] = s
                yHy += s * (gnew[i] - grad[i])
            for i in range(q):
                si = trial[i] - theta[i]
                for j in range(q):
                    sj = trial[j] - theta[j]
                    Hinv[i, j] += (1.0 + yHy / sy) * si * sj / sy - (Hy[i] * sj + si * Hy[j]) / sy
        small = abs(f - fnew) < 1e-14 * (abs(f) + 1.0)
        for j in range(q):
            theta[j] = trial[j]
            grad[j] = gnew[j]
        f = fnew
        mu = munew
        if small and (escaped or not probe):
            break
        if small:
            for j in range(q):
                grad[j] = 0.0
    return mu, f


@njit(cache=True, nogil=True)
def _ma_column(e, s, T, q, ncal, window, refit_every, out):
    """out[t] = MA(q) mean fitted on the errors known at t (all of them, or
    the last ``window``), for every t with at least ``ncal`` errors."""
    theta = np.zeros(q)
    cold = np.zeros(q)
    Hw = np.eye(q)
    Hc = np.eye(q)
    fits = 0
    last = math.nan
    count = 0
    for t in range(s + ncal - 1, T):
        lo = s
        if window > 0 and t - window + 1 > s:
            lo = t - window + 1
        n = t - lo + 1
        if n < q + 3:
            out[t] = math.nan
            continue
        if count % refit_every == 0 or not math.isfinite(last):
            fits += 1
            fw = math.inf
            if fits > 1 and math.isfinite(last):
                last, fw = _ma_fit(e, lo, n, q, theta, Hw, False)
            else:
                last = math.nan
            if fits % 5 == 1 or not math.isfinite(last):
                # a fresh start from zero now and then; keep the better fit
                for j in range(q):
                    cold[j] = 0.0
                    for i in range(q):
                        Hc[i, j] = 1.0 if i == j else 0.0
                mc, fc = _ma_fit(e, lo, n, q, cold, Hc, True)
                if math.isfinite(mc) and (not math.isfinite(last) or fc < fw - 1e-10):
                    last = mc
                    for j in range(q):
                        theta[j] = cold[j]
                        for i in range(q):
                            Hw[i, j] = Hc[i, j]
        out[t] = last
        count += 1


@njit(cache=True, nogil=True, parallel=True)
def ma_means(E, starts, ncal, window, refit_every, MA):
    T, H = E.shape
    for h in prange(2, H + 1):
        if starts[h - 1] < T:
            _ma_column(E[:, h - 1].copy(), starts[h - 1], T, h - 1, ncal, window, refit_every, MA[:, h - 1])


@njit(cache=True, nogil=True)
def _solve(A, b, p):
    """Solve the p x p system in place (partial pivoting). False if singular."""
    for c in range(p):
        piv = c
        big = abs(A[c, c])
        for r in range(c + 1, p):
            if abs(A[r, c]) > big:
                big = abs(A[r, c])
                piv = r
        if big < 1e-12:
            return False
        if piv != c:
            for k in range(p):
                tmp = A[c, k]
                A[c, k] = A[piv, k]
                A[piv, k] = tmp
            tmp = b[c]
            b[c] = b[piv]
            b[piv] = tmp
        for r in range(c + 1, p):
            f = A[r, c] / A[c, c]
            if f != 0.0:
                for k in range(c, p):
                    A[r, k] -= f * A[c, k]
                b[r] -= f * b[c]
    for c in range(p - 1, -1, -1):
        s = b[c]
        for k in range(c + 1, p):
            s -= A[c, k] * b[k]
        b[c] = s / A[c, c]
    return True


@njit(cache=True, nogil=True)
def _add_origin(E, o, h, A, b, x, sign):
    x[0] = 1.0
    for j in range(1, h):
        x[j] = E[o + j, j - 1]
    yv = E[o + h, h - 1]
    for i in range(h):
        for k in range(h):
            A[i, k] += sign * x[i] * x[k]
        b[i] += sign * x[i] * yv


@njit(cache=True, nogil=True)
def scorecast(E, starts, ncal, window, MA, SC):
    """SC[t + h, h - 1] = forecast, made at time t, of the h-step error for
    target t + h. NaN where no scorecast is made."""
    T, H = E.shape
    for h in range(1, H + 1):
        s = starts[h - 1]
        if s >= T:
            continue
        first = -(10**9)  # first origin whose errors exist at every horizon up to h
        for j in range(1, h + 1):
            if starts[j - 1] - j > first:
                first = starts[j - 1] - j
        A = np.zeros((h, h))
        b = np.zeros(h)
        A2 = np.zeros((h, h))
        b2 = np.zeros(h)
        x = np.zeros(h)
        nxt = first
        run = 0.0
        for t in range(s, T):
            run += E[t, h - 1]
            m = t - s + 1
            if h >= 2 and window == 0:
                while nxt <= t - h:
                    _add_origin(E, nxt, h, A, b, x, 1.0)
                    nxt += 1
            if m < ncal:
                continue
            if h == 1:
                if window > 0 and m > window:
                    tot = 0.0
                    for r in range(t - window + 1, t + 1):
                        tot += E[r, 0]
                    SC[t + 1, 0] = tot / window
                else:
                    SC[t + 1, 0] = run / m
                continue
            ok = True
            for j in range(1, h):
                r = t - h + 1 + j
                if r < 0 or not math.isfinite(SC[r, j - 1]):
                    ok = False
                    break
            if not ok or not math.isfinite(MA[t, h - 1]):
                continue
            if window > 0:
                for i in range(h):
                    b[i] = 0.0
                    for k in range(h):
                        A[i, k] = 0.0
                o0 = t - window + 1 - h
                if o0 < first:
                    o0 = first
                for o in range(o0, t - h + 1):
                    _add_origin(E, o, h, A, b, x, 1.0)
            for i in range(h):
                b2[i] = b[i]
                for k in range(h):
                    A2[i, k] = A[i, k]
            if not _solve(A2, b2, h):
                continue
            pred = b2[0]
            for j in range(1, h):
                pred += b2[j] * SC[t - h + 1 + j, j - 1]
            SC[t + h, h - 1] = 0.5 * (MA[t, h - 1] + pred)


@njit(cache=True, nogil=True)
def track(e, s, h, ncal, window, a_half, lr, KI, Csat, integrate, sc, sign, qs):
    """One side of one horizon. qs[r] is the half-width used for target r
    (zero until the first update reaches it)."""
    T = e.shape[0]
    qts = 0.0
    cum = 0.0
    hi = -math.inf
    lo = math.inf
    big = 0.0
    for t in range(s, T):
        v = e[t]
        miss = 1.0 if sign * v > qs[t] else 0.0
        m = t - s + 1
        if abs(v) > big:
            big = abs(v)
        if window > 0 and m > window:
            hi = -math.inf
            lo = math.inf
            for r in range(t - window + 1, t + 1):
                if e[r] > hi:
                    hi = e[r]
                if e[r] < lo:
                    lo = e[r]
        else:
            if v > hi:
                hi = v
            if v < lo:
                lo = v
        lr_t = lr if m <= 1 else lr * (hi - lo)
        qts += lr_t * (miss - a_half)
        integ = 0.0
        if integrate:
            cum += miss
            k = KI if KI >= 0.0 else big
            if m > 1 and k > 0.0:
                arg = (cum - m * a_half) * math.log(m) / (Csat * m)
                if arg >= math.pi / 2:
                    integ = math.inf
                elif arg <= -math.pi / 2:
                    integ = -math.inf
                else:
                    integ = k * math.tan(arg)
        add = 0.0
        if m >= ncal and math.isfinite(sc[t + h]):
            add = sign * sc[t + h]
        qs[t + h] = qts + integ + add
