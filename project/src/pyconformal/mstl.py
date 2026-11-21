"""STL and MSTL seasonal-trend decomposition.

``stl`` is the seasonal-trend decomposition by loess of Cleveland et al.
(1990). ``mstl`` applies it in turn to each seasonal period of a series with
several (Bandara, Hyndman and Bergmeir, 2021), for example the day and the
week in hourly data.

Defaults follow R's ``stl()`` and ``forecast::mstl()``: every loess smoother
is evaluated at every ceil(window / 10)-th point and interpolated in between.
That is the same algorithm as evaluating at every point, and many times
faster. Compiled with numba when it is installed; without it the same code
runs as plain Python, correctly but slowly.

References
----------
Cleveland, Cleveland, McRae and Terpenning (1990). STL: a seasonal-trend
decomposition procedure based on loess. Journal of Official Statistics 6.

Bandara, Hyndman and Bergmeir (2021). MSTL: a seasonal-trend decomposition
algorithm for time series with multiple seasonal patterns. arXiv:2107.13462.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["stl", "mstl", "STLResult", "MSTLResult"]

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        return lambda f: f


# Positions are 1-based inside the kernels, as in the original description:
# the value at position j of a series y is y[j - 1].


@njit(cache=True)
def _est(y, n, length, degree, xs, nleft, nright, w):
    """Local fit at position ``xs`` from positions nleft..nright.
    Returns (value, ok)."""
    h = max(xs - nleft, nright - xs)
    if length > n:
        h += (length - n) // 2
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    for j in range(nleft, nright + 1):
        r = abs(j - xs)
        if r <= h9:
            if r <= h1:
                w[j - 1] = 1.0
            else:
                u = r / h
                v = 1.0 - u * u * u
                w[j - 1] = v * v * v
            a += w[j - 1]
        else:
            w[j - 1] = 0.0
    if a <= 0.0:
        return 0.0, False
    for j in range(nleft, nright + 1):
        w[j - 1] /= a
    if h > 0 and degree > 0:
        a = 0.0
        for j in range(nleft, nright + 1):
            a += w[j - 1] * j
        b = xs - a
        c = 0.0
        for j in range(nleft, nright + 1):
            c += w[j - 1] * (j - a) * (j - a)
        if math.sqrt(c) > 0.001 * (n - 1):
            b /= c
            for j in range(nleft, nright + 1):
                w[j - 1] *= b * (j - a) + 1.0
    s = 0.0
    for j in range(nleft, nright + 1):
        s += w[j - 1] * y[j - 1]
    return s, True


@njit(cache=True)
def _loess(y, n, length, degree, jump, ys, w):
    """Loess smooth of y[0:n] into ys[0:n], evaluated every ``jump`` points
    and linearly interpolated in between."""
    if n < 2:
        ys[0] = y[0]
        return
    nj = min(jump, n - 1)
    nleft = 1
    nright = n
    if length >= n:
        for i in range(1, n + 1, nj):
            v, ok = _est(y, n, length, degree, i, nleft, nright, w)
            ys[i - 1] = v if ok else y[i - 1]
    elif nj == 1:
        nsh = (length + 1) // 2
        nright = length
        for i in range(1, n + 1):
            if i > nsh and nright != n:
                nleft += 1
                nright += 1
            v, ok = _est(y, n, length, degree, i, nleft, nright, w)
            ys[i - 1] = v if ok else y[i - 1]
    else:
        nsh = (length + 1) // 2
        for i in range(1, n + 1, nj):
            if i < nsh:
                nleft = 1
                nright = length
            elif i >= n - nsh + 1:
                nleft = n - length + 1
                nright = n
            else:
                nleft = i - nsh + 1
                nright = length + i - nsh
            v, ok = _est(y, n, length, degree, i, nleft, nright, w)
            ys[i - 1] = v if ok else y[i - 1]
    if nj != 1:
        for i in range(1, n - nj + 1, nj):
            delta = (ys[i + nj - 1] - ys[i - 1]) / nj
            for j in range(i + 1, i + nj):
                ys[j - 1] = ys[i - 1] + delta * (j - i)
        k = ((n - 1) // nj) * nj + 1
        if k != n:
            v, ok = _est(y, n, length, degree, n, nleft, nright, w)
            ys[n - 1] = v if ok else y[n - 1]
            if k != n - 1:
                delta = (ys[n - 1] - ys[k - 1]) / (n - k)
                for j in range(k + 1, n):
                    ys[j - 1] = ys[k - 1] + delta * (j - k)


@njit(cache=True)
def _moving_average(x, n, length, out):
    newn = n - length + 1
    v = 0.0
    for i in range(length):
        v += x[i]
    out[0] = v / length
    for j in range(1, newn):
        v += x[length + j - 1] - x[j - 1]
        out[j] = v / length


@njit(cache=True)
def _stl(y, period, ns, nt, nl, sdeg, tdeg, ldeg, sjump, tjump, ljump, inner, seasonal, trend):
    n = y.shape[0]
    cyc = np.empty(n + 2 * period)  # smoothed cycle-subseries, one period of padding each side
    work = np.empty(n + 2 * period)
    work2 = np.empty(n + 2 * period)
    low = np.empty(n)
    w = np.empty(n)
    kmax = (n - 1) // period + 1
    sub = np.empty(kmax)
    sub_s = np.empty(kmax + 2)
    for i in range(n):
        trend[i] = 0.0
    for _ in range(inner):
        for i in range(n):
            work[i] = y[i] - trend[i]
        # smooth each cycle-subseries, with one value extrapolated at each end
        for j in range(period):
            k = (n - j - 1) // period + 1
            for i in range(k):
                sub[i] = work[i * period + j]
            _loess(sub, k, ns, sdeg, sjump, sub_s[1:], w)
            nright = min(ns, k)
            v, ok = _est(sub, k, ns, sdeg, 0, 1, nright, w)
            sub_s[0] = v if ok else sub_s[1]
            nleft = max(1, k - ns + 1)
            v, ok = _est(sub, k, ns, sdeg, k + 1, nleft, k, w)
            sub_s[k + 1] = v if ok else sub_s[k]
            for m in range(k + 2):
                cyc[m * period + j] = sub_s[m]
        # low-pass filter: moving averages of length period, period and 3, then loess
        _moving_average(cyc, n + 2 * period, period, work)
        _moving_average(work, n + period + 1, period, work2)
        _moving_average(work2, n + 2, 3, work)
        _loess(work, n, nl, ldeg, ljump, low, w)
        for i in range(n):
            seasonal[i] = cyc[period + i] - low[i]
            work[i] = y[i] - seasonal[i]
        _loess(work, n, nt, tdeg, tjump, trend, w)


@njit(cache=True)
def _mstl(y, periods, windows, sdeg, iterate, full, inner, seasonal, trend):
    """Seasonal components (one row per period) and trend, as forecast::mstl."""
    n = y.shape[0]
    des = y.copy()
    comp = np.empty(n)
    for i in range(periods.shape[0]):
        for t in range(n):
            seasonal[i, t] = 0.0
    for _ in range(iterate):
        for i in range(periods.shape[0]):
            p = periods[i]
            ns = windows[i]
            nt = int(math.ceil(1.5 * p / (1.0 - 1.5 / ns)))
            nt += 1 - nt % 2
            nl = p + 1 - p % 2
            for t in range(n):
                des[t] += seasonal[i, t]
            sj = 1 if full else int(math.ceil(ns / 10.0))
            tj = 1 if full else int(math.ceil(nt / 10.0))
            lj = 1 if full else int(math.ceil(nl / 10.0))
            _stl(des, p, ns, nt, nl, sdeg, 1, 1, sj, tj, lj, inner, comp, trend)
            for t in range(n):
                seasonal[i, t] = comp[t]
                des[t] -= comp[t]


def _odd(v):
    v = int(v)
    return v + 1 - v % 2


def _series(y):
    y = np.ascontiguousarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("y must be finite. Fill missing values first.")
    return y


@dataclass
class STLResult:
    observed: np.ndarray
    seasonal: np.ndarray
    trend: np.ndarray

    @property
    def remainder(self):
        return self.observed - self.seasonal - self.trend


@dataclass
class MSTLResult:
    observed: np.ndarray
    seasonal: dict  # period -> component
    trend: np.ndarray

    @property
    def remainder(self):
        return self.observed - self.trend - sum(self.seasonal.values())

    @property
    def seasonal_total(self):
        return sum(self.seasonal.values())


def stl(
    y,
    period: int,
    seasonal: int = 7,
    trend: int | None = None,
    low_pass: int | None = None,
    seasonal_deg: int = 0,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    seasonal_jump: int | None = None,
    trend_jump: int | None = None,
    low_pass_jump: int | None = None,
    inner_iter: int = 2,
) -> STLResult:
    """Seasonal-trend decomposition by loess (no robustness iterations).

    Windows must be odd. Defaults are those of R's ``stl``: local constant
    seasonal fit, trend window ``nextodd(ceil(1.5 * period / (1 - 1.5 /
    seasonal)))``, low-pass window ``nextodd(period)``, and jumps of
    ``ceil(window / 10)``. Set the three jumps to 1 to evaluate every point.
    """
    y = _series(y)
    period, ns = int(period), int(seasonal)
    if period < 2:
        raise ValueError("period must be at least 2.")
    if y.shape[0] < 2 * period:
        raise ValueError("The series must hold at least two full periods.")
    nt = _odd(math.ceil(1.5 * period / (1.0 - 1.5 / ns))) if trend is None else int(trend)
    nl = _odd(period) if low_pass is None else int(low_pass)
    for name, v in (("seasonal", ns), ("trend", nt), ("low_pass", nl)):
        if v < 3 or v % 2 == 0:
            raise ValueError(f"{name} must be an odd number of at least 3.")
    for name, v in (("seasonal_deg", seasonal_deg), ("trend_deg", trend_deg), ("low_pass_deg", low_pass_deg)):
        if v not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1.")
    sj = math.ceil(ns / 10) if seasonal_jump is None else int(seasonal_jump)
    tj = math.ceil(nt / 10) if trend_jump is None else int(trend_jump)
    lj = math.ceil(nl / 10) if low_pass_jump is None else int(low_pass_jump)
    if min(sj, tj, lj, int(inner_iter)) < 1:
        raise ValueError("Jumps and inner_iter must be at least 1.")
    s = np.empty_like(y)
    t = np.empty_like(y)
    _stl(y, period, ns, nt, nl, int(seasonal_deg), int(trend_deg), int(low_pass_deg), sj, tj, lj, int(inner_iter), s, t)
    return STLResult(y, s, t)


def mstl(
    y, periods, iterate: int = 2, seasonal_windows=None, seasonal_deg: int = 0, jumps: bool = True, inner_iter: int = 2
) -> MSTLResult:
    """MSTL decomposition for one or more seasonal periods.

    Same procedure and defaults as R's ``forecast::mstl``: periods in
    increasing order, seasonal windows 11, 15, 19, ..., two passes. Periods
    that do not fit twice in the series are dropped. ``jumps=False``
    evaluates every loess fit at every point (slower, almost identical).
    ``mstl(y, periods, jumps=False, inner_iter=5)`` gives
    what statsforecast's ``mstl`` computes.

    Returns an :class:`MSTLResult` with ``seasonal`` (a dict, one component
    per period), ``trend`` and ``remainder``.
    """
    y = _series(y)
    ps = sorted({int(p) for p in np.atleast_1d(periods)})
    if any(p < 2 for p in ps):
        raise ValueError("Every period must be at least 2.")
    ps = [p for p in ps if y.shape[0] > 2 * p]
    if not ps:
        raise ValueError("No period fits twice in the series.")
    if seasonal_windows is None:
        sw = [7 + 4 * (i + 1) for i in range(len(ps))]
    else:
        sw = list(np.atleast_1d(seasonal_windows))
        sw = sw * len(ps) if len(sw) == 1 else sw[: len(ps)]
        if len(sw) != len(ps) or any(int(v) < 3 or int(v) % 2 == 0 for v in sw):
            raise ValueError("seasonal_windows must give one odd window of at least 3 per period.")
    if seasonal_deg not in (0, 1):
        raise ValueError("seasonal_deg must be 0 or 1.")
    it = 1 if len(ps) == 1 else max(1, int(iterate))
    seas = np.empty((len(ps), y.shape[0]))
    trend = np.empty_like(y)
    if int(inner_iter) < 1:
        raise ValueError("inner_iter must be at least 1.")
    _mstl(
        y, np.array(ps, dtype=np.int64), np.array(sw, dtype=np.int64), int(seasonal_deg), it, not jumps,
        int(inner_iter), seas, trend,
    )
    return MSTLResult(y, {p: seas[i] for i, p in enumerate(ps)}, trend)
