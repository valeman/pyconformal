"""Kernel for adaptive conformal inference over a whole stream.

The calibration set changes at every step and the requested rank changes
with the working alpha, so each step needs "the k-th smallest of a changing
set". All scores are known when a stream is replayed, so they are ranked
once and kept in a Fenwick (binary indexed) tree of counts. Insert, delete
and k-th-smallest are then O(log N), and a stream of T steps costs
O((n + T) log(n + T)) in total.
"""

from __future__ import annotations

import math

# Guards against (n + 1) * (1 - alpha) landing a hair above an integer in
# floating point (for example 21.000000000000004 instead of 21).
_RANK_TOL = 1e-9


def _aci_py(vals, rank, n0, alpha, gamma, alpha_t, window, tree, q, err, alphas):
    """Replay a stream.

    vals   : all scores sorted ascending (calibration and stream together)
    rank   : rank[i] = position in ``vals`` of the i-th score in arrival
             order (first the n0 calibration scores, then the stream)
    window : 0 for an expanding calibration set, else its maximum size
    tree   : zeroed Fenwick tree of length N + 1
    q, err, alphas : outputs of length T
    Returns the working alpha after the last step.
    """
    N = len(vals)
    T = N - n0
    top = 1
    while top * 2 <= N:
        top *= 2

    head = 0  # arrival index of the oldest score still in the set
    if window > 0 and n0 > window:
        head = n0 - window
    n = 0
    for i in range(head, n0):
        j = rank[i] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1

    for t in range(T):
        alphas[t] = alpha_t
        s = vals[rank[n0 + t]]
        if alpha_t <= 0.0:
            thr = math.inf
        else:
            k = math.ceil((n + 1) * (1.0 - alpha_t) - _RANK_TOL)
            if k > n:
                thr = math.inf
            elif k < 1:
                thr = -math.inf
            else:
                pos = 0
                step = top
                while step > 0:
                    nxt = pos + step
                    if nxt <= N and tree[nxt] < k:
                        pos = nxt
                        k -= tree[nxt]
                    step //= 2
                thr = vals[pos]
        q[t] = thr
        miss = s > thr
        err[t] = miss
        alpha_t = alpha_t + gamma * (alpha - (1.0 if miss else 0.0))

        j = rank[n0 + t] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
        if window > 0 and n > window:
            j = rank[head] + 1
            while j <= N:
                tree[j] -= 1
                j += j & (-j)
            head += 1
            n -= 1
    return alpha_t


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit

    _aci_nb = njit(cache=True, nogil=True)(_aci_py)
    HAVE_NUMBA = True
except Exception:
    _aci_nb = None
    HAVE_NUMBA = False


def aci_replay(cal, stream, alpha, gamma, alpha_t, window, use_numba=None):
    """Run ACI over ``stream`` starting from calibration scores ``cal``.

    Returns (thresholds, errors, alphas, final_alpha_t).
    """
    import numpy as np

    n0 = cal.shape[0]
    T = stream.shape[0]
    allv = np.concatenate([cal, stream])
    order = np.argsort(allv, kind="stable")
    rank = np.empty(n0 + T, dtype=np.int64)
    rank[order] = np.arange(n0 + T)
    vals = allv[order]
    if use_numba is None:
        use_numba = HAVE_NUMBA
    if use_numba:
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not available.")
        tree = np.zeros(n0 + T + 1, dtype=np.int64)
        q = np.empty(T)
        err = np.empty(T, dtype=np.bool_)
        alphas = np.empty(T)
        final = _aci_nb(vals, rank, n0, alpha, gamma, alpha_t, window, tree, q, err, alphas)
        return q, err, alphas, float(final)
    tree = [0] * (n0 + T + 1)
    q = [0.0] * T
    err = [False] * T
    alphas = [0.0] * T
    final = _aci_py(
        vals.tolist(), rank.tolist(), n0, alpha, gamma, alpha_t, window, tree, q, err, alphas
    )
    return np.asarray(q), np.asarray(err, dtype=bool), np.asarray(alphas), float(final)


def _dtaci_step(a, w, gammas, alpha, eta, sigma, r, n):
    """One DtACI update, in place. ``r`` of the ``n`` past scores are strictly
    below the new score. Returns the combined working level for the next step.
    """
    K = len(a)
    beta = 1.0 - r / (n + 1.0)  # the levels below beta would have covered
    tot = 0.0
    for i in range(K):
        d = beta - a[i]
        loss = alpha * d - (d if d < 0.0 else 0.0)
        w[i] = w[i] * math.exp(-eta * loss)
        tot += w[i]
    for i in range(K):
        w[i] = (1.0 - sigma) * w[i] / tot + sigma / K
    for i in range(K):
        # did expert i miss? Same rank rule as the threshold itself.
        miss = False
        if a[i] > 0.0:
            k = math.ceil((n + 1) * (1.0 - a[i]) - _RANK_TOL)
            if k <= n:
                miss = k < 1 or r >= k
        a[i] = a[i] + gammas[i] * (alpha - (1.0 if miss else 0.0))
    out = 0.0
    for i in range(K):
        out += w[i] * a[i]
    return out


def _dtaci_py(vals, rank, low, n0, alpha, gammas, a, w, eta, sigma, alpha_t, window, tree, q, err, alphas):
    """As ``_aci_py`` with several step sizes. ``low[i]`` is the number of
    all values strictly below the i-th arrival's value."""
    N = len(vals)
    T = N - n0
    top = 1
    while top * 2 <= N:
        top *= 2
    head = 0
    if window > 0 and n0 > window:
        head = n0 - window
    n = 0
    for i in range(head, n0):
        j = rank[i] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
    for t in range(T):
        alphas[t] = alpha_t
        s = vals[rank[n0 + t]]
        if alpha_t <= 0.0:
            thr = math.inf
        else:
            k = math.ceil((n + 1) * (1.0 - alpha_t) - _RANK_TOL)
            if k > n:
                thr = math.inf
            elif k < 1:
                thr = -math.inf
            else:
                pos = 0
                step = top
                while step > 0:
                    nxt = pos + step
                    if nxt <= N and tree[nxt] < k:
                        pos = nxt
                        k -= tree[nxt]
                    step //= 2
                thr = vals[pos]
        q[t] = thr
        err[t] = s > thr
        r = 0  # past scores strictly below s
        j = low[n0 + t]
        while j > 0:
            r += tree[j]
            j -= j & (-j)
        alpha_t = _dtaci_step(a, w, gammas, alpha, eta, sigma, r, n)
        j = rank[n0 + t] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
        if window > 0 and n > window:
            j = rank[head] + 1
            while j <= N:
                tree[j] -= 1
                j += j & (-j)
            head += 1
            n -= 1
    return alpha_t


if HAVE_NUMBA:  # pragma: no cover
    _dtaci_step_nb = njit(cache=True, nogil=True)(_dtaci_step)
    _g = dict(globals())
    _g["_dtaci_step"] = _dtaci_step_nb
    import types as _types

    _dtaci_nb = njit(cache=True, nogil=True)(
        _types.FunctionType(_dtaci_py.__code__, _g, "_dtaci_py", _dtaci_py.__defaults__)
    )


def dtaci_replay(cal, stream, alpha, gammas, a, w, eta, sigma, alpha_t, window, use_numba=None):
    """Run DtACI over ``stream``. Updates ``a`` and ``w`` in place.
    Returns (thresholds, errors, alphas, final working level)."""
    import numpy as np

    n0 = cal.shape[0]
    T = stream.shape[0]
    allv = np.concatenate([cal, stream])
    order = np.argsort(allv, kind="stable")
    rank = np.empty(n0 + T, dtype=np.int64)
    rank[order] = np.arange(n0 + T)
    vals = allv[order]
    low = np.searchsorted(vals, allv, side="left").astype(np.int64)
    if use_numba is None:
        use_numba = HAVE_NUMBA
    tree = np.zeros(n0 + T + 1, dtype=np.int64)
    q = np.empty(T)
    err = np.empty(T, dtype=np.bool_)
    alphas = np.empty(T)
    fn = _dtaci_nb if use_numba else _dtaci_py
    final = fn(vals, rank, low, n0, alpha, gammas, a, w, eta, sigma, alpha_t, window, tree, q, err, alphas)
    return q, err, alphas, float(final)


def _nexcp_py(vals, rank, n0, alpha, lg, tree, q, err):
    """Replay non-exchangeable conformal prediction with geometric weights.

    The score that arrived ``d`` steps before the one being predicted has
    weight ``exp(-lg * d)``; the point being predicted has weight 1 and sits
    at +infinity. Weights are stored relative to a moving origin (so they
    never overflow) in a Fenwick tree over score ranks, which gives the
    weighted quantile in O(log N) per step.
    """
    N = len(vals)
    top = 1
    while top * 2 <= N:
        top *= 2
    base = 0
    total = 0.0
    for i in range(N):
        if i >= n0:
            t = i - n0
            w_new = math.exp((i - base) * lg)  # weight of the point being predicted
            need = (1.0 - alpha) * (total + w_new) * (1.0 - 1e-12)
            if total < need:
                thr = math.inf
            else:
                pos = 0
                rem = need
                step = top
                while step > 0:
                    nxt = pos + step
                    if nxt <= N and tree[nxt] < rem:
                        pos = nxt
                        rem -= tree[nxt]
                    step //= 2
                thr = vals[pos] if pos < N else math.inf
            q[t] = thr
            err[t] = vals[rank[i]] > thr
        w = math.exp((i - base) * lg)
        j = rank[i] + 1
        while j <= N:
            tree[j] += w
            j += j & (-j)
        total += w
        if (i + 1 - base) * lg > 300.0:  # move the origin before weights get large
            f = math.exp(-(i + 1 - base) * lg)
            for j in range(N + 1):
                tree[j] *= f
            total *= f
            base = i + 1


if HAVE_NUMBA:  # pragma: no cover
    _nexcp_nb = njit(cache=True, nogil=True)(_nexcp_py)


def nexcp_replay(cal, stream, alpha, rho, use_numba=None):
    import numpy as np

    n0, T = cal.shape[0], stream.shape[0]
    allv = np.concatenate([cal, stream])
    order = np.argsort(allv, kind="stable")
    rank = np.empty(n0 + T, dtype=np.int64)
    rank[order] = np.arange(n0 + T)
    tree = np.zeros(n0 + T + 1)
    q = np.empty(T)
    err = np.empty(T, dtype=np.bool_)
    if use_numba is None:
        use_numba = HAVE_NUMBA
    (_nexcp_nb if use_numba else _nexcp_py)(allv[order], rank, n0, alpha, -math.log(rho), tree, q, err)
    return q, err


# ------------------------------------------------------------------ AgACI


def _boa_weights(eta_inv2, R_reg, w):
    """Bernstein online aggregation weights (Wintenberger, 2017), as in the
    R package opera used by the AgACI authors. Uniform prior. Experts with
    no regret history yet keep the prior weight."""
    K = len(w)
    best = -math.inf
    any_ = False
    for k in range(K):
        if eta_inv2[k] > 0.0:
            v = -0.5 * math.log(eta_inv2[k]) + R_reg[k] / math.sqrt(eta_inv2[k])
            w[k] = v
            any_ = True
            if v > best:
                best = v
    if not any_:
        for k in range(K):
            w[k] = 1.0 / K
        return
    tot = 0.0
    m = 0
    for k in range(K):
        if eta_inv2[k] > 0.0:
            w[k] = math.exp(w[k] - best)
            tot += w[k]
            m += 1
    for k in range(K):
        if eta_inv2[k] > 0.0:
            w[k] = w[k] / tot * (m / K)
        else:
            w[k] = 1.0 / K


def _agaci_combine(alpha, expert_q, eta_inv2, R_reg, w, cap):
    """Aggregated threshold from the experts' thresholds; an expert that asks
    for an infinite threshold is counted at ``cap``, the largest score seen."""
    _boa_weights(eta_inv2, R_reg, w)
    out = 0.0
    for k in range(len(w)):
        if expert_q[k] > cap:
            expert_q[k] = cap
        if expert_q[k] < 0.0:
            expert_q[k] = 0.0
        out += w[k] * expert_q[k]
    return out


def _agaci_learn(alpha, gammas, a, expert_q, eta_inv2, R, R_reg, s, pred, raw_miss):
    """After seeing score ``s``: BOA regret update at pinball level
    ``1 - alpha``, then each expert's own ACI update."""
    g = (1.0 if s < pred else 0.0) - (1.0 - alpha)
    for k in range(len(a)):
        r = g * (pred - expert_q[k])
        eta_inv2[k] += 2.2 * r * r
        if eta_inv2[k] > 0.0:
            R_reg[k] += r - r * r / math.sqrt(eta_inv2[k])
        R[k] += r
        a[k] = a[k] + gammas[k] * (alpha - (1.0 if raw_miss[k] else 0.0))


def _agaci_py(vals, rank, n0, alpha, gammas, a, eta_inv2, R, R_reg, w, cap, window, tree, q, err, eq, miss):
    N = len(vals)
    T = N - n0
    K = len(a)
    top = 1
    while top * 2 <= N:
        top *= 2
    head = 0
    if window > 0 and n0 > window:
        head = n0 - window
    n = 0
    for i in range(head, n0):
        j = rank[i] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
    for t in range(T):
        s = vals[rank[n0 + t]]
        for k in range(K):  # each expert's own threshold
            if a[k] <= 0.0:
                thr = math.inf
            else:
                kk = math.ceil((n + 1) * (1.0 - a[k]) - _RANK_TOL)
                if kk > n:
                    thr = math.inf
                elif kk < 1:
                    thr = -math.inf
                else:
                    pos = 0
                    step = top
                    while step > 0:
                        nxt = pos + step
                        if nxt <= N and tree[nxt] < kk:
                            pos = nxt
                            kk -= tree[nxt]
                        step //= 2
                    thr = vals[pos]
            miss[k] = s > thr
            eq[k] = thr
        if n == 0:
            pred = math.inf
        else:
            pred = _agaci_combine(alpha, eq, eta_inv2, R_reg, w, cap)
        q[t] = pred
        err[t] = s > pred
        if n > 0:
            _agaci_learn(alpha, gammas, a, eq, eta_inv2, R, R_reg, s, pred, miss)
        else:
            for k in range(K):
                a[k] = a[k] + gammas[k] * (alpha - (1.0 if miss[k] else 0.0))
        if s > cap:
            cap = s
        j = rank[n0 + t] + 1
        while j <= N:
            tree[j] += 1
            j += j & (-j)
        n += 1
        if window > 0 and n > window:
            j = rank[head] + 1
            while j <= N:
                tree[j] -= 1
                j += j & (-j)
            head += 1
            n -= 1
    return cap


if HAVE_NUMBA:  # pragma: no cover
    import types as _types

    def _jit_with(fn, **deps):
        g = dict(globals())
        g.update(deps)
        return njit(cache=True, nogil=True)(_types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__))

    _boa_weights_nb = njit(cache=True, nogil=True)(_boa_weights)
    _agaci_combine_nb = _jit_with(_agaci_combine, _boa_weights=_boa_weights_nb)
    _agaci_learn_nb = njit(cache=True, nogil=True)(_agaci_learn)
    _agaci_nb = _jit_with(_agaci_py, _agaci_combine=_agaci_combine_nb, _agaci_learn=_agaci_learn_nb)


def agaci_replay(cal, stream, alpha, gammas, a, eta_inv2, R, R_reg, cap, window, use_numba=None):
    """Run AgACI over ``stream``; state arrays are updated in place.
    Returns (thresholds, errors, new cap)."""
    import numpy as np

    n0, T = cal.shape[0], stream.shape[0]
    allv = np.concatenate([cal, stream])
    order = np.argsort(allv, kind="stable")
    rank = np.empty(n0 + T, dtype=np.int64)
    rank[order] = np.arange(n0 + T)
    K = len(a)
    tree = np.zeros(n0 + T + 1, dtype=np.int64)
    q = np.empty(T)
    err = np.empty(T, dtype=np.bool_)
    if use_numba is None:
        use_numba = HAVE_NUMBA
    fn = _agaci_nb if use_numba else _agaci_py
    cap = fn(allv[order], rank, n0, alpha, gammas, a, eta_inv2, R, R_reg, np.empty(K), cap, window, tree, q, err,
             np.empty(K), np.empty(K, dtype=np.bool_))
    return q, err, float(cap)


# -------------------------------------------------------------------- PID


def _pid_py(scores, alpha, lr, KI, Csat, burn_in, proportional, state, recent, q, err):
    """Quantile tracking with optional integrator over a stream.

    ``state`` = [q, integrator, miss_sum, t, number of recent scores held];
    ``recent`` is a ring buffer of the last ``burn_in`` scores. Both are
    updated in place, so a replay can be continued."""
    qt = state[0]
    integ = state[1]
    miss_sum = state[2]
    t = int(state[3])
    held = int(state[4])
    for i in range(len(scores)):
        s = scores[i]
        thr = qt + integ
        q[i] = thr
        if proportional and t > 0:
            lo = recent[0]
            hi = recent[0]
            for j in range(1, held):
                v = recent[j]
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
            lr_t = lr * (hi - lo)
        else:
            lr_t = lr
        covered = thr >= s
        err[i] = not covered
        if KI > 0.0:
            x = (miss_sum - t * alpha) * math.log(t + 1) / (Csat * (t + 1))
            if x >= math.pi / 2:
                integ = math.inf
            elif x <= -math.pi / 2:
                integ = -math.inf
            else:
                integ = KI * math.tan(x)
        else:
            integ = 0.0
        if covered:
            qt = qt - lr_t * alpha
        else:
            qt = qt + lr_t * (1.0 - alpha)
            miss_sum += 1.0
        recent[t % burn_in] = s
        if held < burn_in:
            held += 1
        t += 1
    state[0] = qt
    state[1] = integ
    state[2] = miss_sum
    state[3] = t
    state[4] = held


if HAVE_NUMBA:  # pragma: no cover
    _pid_nb = njit(cache=True, nogil=True)(_pid_py)


def pid_replay(scores, alpha, lr, KI, Csat, burn_in, proportional, state, recent, use_numba=None):
    import numpy as np

    q = np.empty(scores.shape[0])
    err = np.empty(scores.shape[0], dtype=np.bool_)
    if use_numba is None:
        use_numba = HAVE_NUMBA
    (_pid_nb if use_numba else _pid_py)(scores, alpha, lr, KI, Csat, burn_in, proportional, state, recent, q, err)
    return q, err

