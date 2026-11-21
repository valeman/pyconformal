"""Quantile regression forest.

A random forest that predicts conditional quantiles (Meinshausen, 2006): a
new sample is dropped down every tree, the training outcomes in the leaves it
reaches are pooled with weight 1 / (leaf size) per tree, and quantiles are
read from that weighted sample. Any number of quantile levels costs one pass.

Written for speed: features are binned once into at most 256 levels, split
search works on histograms, trees are grown in parallel, and prediction runs
in parallel over samples. Requires numba (``pip install pyconformal[fast]``).

Reference
---------
Meinshausen (2006). Quantile regression forests. JMLR 7, 983-999.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QuantileForest"]

try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        return lambda f: f

    prange = range


@njit(cache=True)
def _next(s):
    # xorshift64*: each tree has its own stream, so results do not depend on
    # the number of threads
    s ^= s >> np.uint64(12)
    s ^= s << np.uint64(25)
    s ^= s >> np.uint64(27)
    return s, s * np.uint64(2685821657736338717)


@njit(cache=True)
def _below(s, m):
    s, v = _next(s)
    return s, np.int64(v >> np.uint64(11)) % m


@njit(cache=True)
def _build_tree(Xb, y, seed, n_draw, bootstrap, mtry, min_leaf, max_depth, feat, thr, a, b, idx):
    """Grow one tree. Node ``j`` is a leaf when ``feat[j] < 0``; then
    ``idx[a[j] : a[j] + b[j]]`` are its training samples. Otherwise samples
    with ``Xb[:, feat[j]] <= thr[j]`` go to node ``a[j]``, the rest to
    ``a[j] + 1``."""
    n, d = Xb.shape
    s = seed
    if bootstrap:
        for i in range(n_draw):
            s, r = _below(s, n)
            idx[i] = r
    else:  # n_draw distinct samples
        perm = np.arange(n)
        for i in range(n_draw):
            s, r = _below(s, n - i)
            perm[i], perm[i + r] = perm[i + r], perm[i]
            idx[i] = perm[i]
    feats = np.arange(d)
    cnt = np.zeros(256, dtype=np.int64)
    sm = np.zeros(256)
    cap = feat.shape[0]
    st_start = np.empty(cap, dtype=np.int64)
    st_end = np.empty(cap, dtype=np.int64)
    st_node = np.empty(cap, dtype=np.int64)
    st_depth = np.empty(cap, dtype=np.int64)
    st_start[0], st_end[0], st_node[0], st_depth[0] = 0, n_draw, 0, 0
    top = 1
    n_nodes = 1
    while top > 0:
        top -= 1
        start, end, node, depth = st_start[top], st_end[top], st_node[top], st_depth[top]
        m = end - start
        total = 0.0
        ymin = np.inf
        ymax = -np.inf
        for k in range(start, end):
            v = y[idx[k]]
            total += v
            if v < ymin:
                ymin = v
            if v > ymax:
                ymax = v
        best_f = -1
        best_bin = 0
        best = -np.inf
        if m >= 2 * min_leaf and depth < max_depth and ymin < ymax:
            tried = 0
            for j in range(d):
                s, r = _below(s, d - j)
                feats[j], feats[j + r] = feats[j + r], feats[j]
                f = feats[j]
                lo = 255
                hi = 0
                for k in range(start, end):
                    i = idx[k]
                    c = Xb[i, f]
                    cnt[c] += 1
                    sm[c] += y[i]
                    if c < lo:
                        lo = c
                    if c > hi:
                        hi = c
                nl = 0
                sl = 0.0
                for c in range(lo, hi):
                    nl += cnt[c]
                    sl += sm[c]
                    cnt[c] = 0
                    sm[c] = 0.0
                    if nl >= min_leaf and m - nl >= min_leaf:
                        gain = sl * sl / nl + (total - sl) * (total - sl) / (m - nl)
                        if gain > best:
                            best = gain
                            best_f = f
                            best_bin = c
                cnt[hi] = 0
                sm[hi] = 0.0
                if lo < hi:  # a feature that is constant here does not count
                    tried += 1
                    if tried >= mtry:
                        break
        if best_f < 0:
            feat[node] = -1
            a[node] = start
            b[node] = m
            continue
        i = start
        j = end - 1
        while i <= j:
            if Xb[idx[i], best_f] <= best_bin:
                i += 1
            else:
                idx[i], idx[j] = idx[j], idx[i]
                j -= 1
        feat[node] = best_f
        thr[node] = best_bin
        a[node] = n_nodes
        b[node] = 0
        st_start[top], st_end[top], st_node[top], st_depth[top] = start, i, n_nodes, depth + 1
        top += 1
        st_start[top], st_end[top], st_node[top], st_depth[top] = i, end, n_nodes + 1, depth + 1
        top += 1
        n_nodes += 2
    return n_nodes


@njit(parallel=True, cache=True)
def _build_forest(Xb, y, seeds, n_draw, bootstrap, mtry, min_leaf, max_depth, feat, thr, a, b, idx, n_nodes):
    for t in prange(seeds.shape[0]):
        n_nodes[t] = _build_tree(
            Xb, y, seeds[t], n_draw, bootstrap, mtry, min_leaf, max_depth, feat[t], thr[t], a[t], b[t], idx[t]
        )


@njit(parallel=True, cache=True)
def _predict(Xb, ys, nodes, b, rk, levels, out_q, out_mean, inbag, block):
    """Weighted quantiles (levels ascending) and mean for every row.

    ``rk`` holds, for every tree, the rank of each training outcome among all
    outcomes, and ``ys`` the outcomes in increasing order. The pooled leaf
    members of a row are therefore integers, which are put in order with a
    bucket sort in linear time instead of a comparison sort.
    """
    n = Xb.shape[0]
    T = nodes.shape[0]
    for blk in prange((n + block - 1) // block):
        first = blk * block
        m = min(block, n - first)
        # Find every leaf first, one tree at a time over the whole block, so
        # the upper levels of each tree stay in the processor cache. A node
        # is one 64-bit word: child or leaf start in the low 32 bits, then
        # the split bin (8 bits), then the feature + 1 (0 marks a leaf).
        # All rows of the block step down the tree together, level by level:
        # the steps of different rows do not depend on each other, so the
        # processor overlaps them instead of waiting on one row at a time.
        leaf = np.empty((m, T), dtype=np.int64)
        cur = np.empty(m, dtype=np.int64)
        act = np.empty(m, dtype=np.int64)
        for t in range(T):
            tree = nodes[t]
            cur[:] = 0
            for r in range(m):
                act[r] = r
            left = m
            while left > 0:  # only rows that have not reached a leaf yet
                k = 0
                for j in range(left):
                    r = act[j]
                    w = tree[cur[r]]
                    f = w >> 40
                    if f != 0:
                        cur[r] = (w & 0xFFFFFFFF) + (Xb[first + r, f - 1] > ((w >> 32) & 0xFF))
                        act[k] = r
                        k += 1
                left = k
            if inbag.shape[0] == 0:
                for r in range(m):
                    leaf[r, t] = cur[r]
            else:  # out-of-bag: a tree that trained on the row does not vote
                for r in range(m):
                    leaf[r, t] = -1 if inbag[t, first + r] else cur[r]
        starts = np.empty(T, dtype=np.int64)
        sizes = np.empty(T, dtype=np.int64)
        cap = 2048
        pos = np.empty(cap + 1, dtype=np.int64)
        sr = np.empty(cap, dtype=np.int64)
        sw = np.empty(cap)
        for r in range(m):
            need = 0
            for t in range(T):
                if leaf[r, t] >= 0:
                    need += b[t, leaf[r, t]]
            if need > cap:  # grow the work buffers, rarely
                cap = 2 * need
                pos = np.empty(cap + 1, dtype=np.int64)
                sr = np.empty(cap, dtype=np.int64)
                sw = np.empty(cap)
            _pool(first + r, leaf[r], ys, nodes, b, rk, levels, out_q, out_mean, starts, sizes, pos, sr, sw)


@njit(cache=True)
def _pool(i, leaf, ys, nodes, b, rk, levels, out_q, out_mean, starts, sizes, pos, sr, sw):
        # starts, sizes, pos, sr and sw are work buffers owned by the caller
        T = nodes.shape[0]
        nq = levels.shape[0]
        total = 0
        rmin = np.int64(2**62)
        rmax = np.int64(-1)
        mean = 0.0
        used = 0
        for t in range(T):
            if leaf[t] >= 0:
                used += 1
        if used == 0:  # the row was in every tree's sample
            out_mean[i] = np.nan
            for qi in range(nq):
                out_q[i, qi] = np.nan
            return
        for t in range(T):
            if leaf[t] < 0:
                starts[t] = 0
                sizes[t] = 0
                continue
            s0 = nodes[t, leaf[t]] & 0xFFFFFFFF
            c = b[t, leaf[t]]
            starts[t] = s0
            sizes[t] = c
            total += c
            wt = 1.0 / (c * used)
            for j in range(s0, s0 + c):
                r = rk[t, j]
                mean += ys[r] * wt
                if r < rmin:
                    rmin = r
                if r > rmax:
                    rmax = r
        out_mean[i] = mean
        if nq == 0:
            return
        span = rmax - rmin + 1
        for k in range(total + 1):
            pos[k] = 0
        for t in range(T):
            for j in range(starts[t], starts[t] + sizes[t]):
                pos[((rk[t, j] - rmin) * total) // span + 1] += 1
        for k in range(total):
            pos[k + 1] += pos[k]
        for t in range(T):
            if sizes[t] == 0:
                continue
            wt = 1.0 / (sizes[t] * used)
            for j in range(starts[t], starts[t] + sizes[t]):
                r = rk[t, j]
                k = ((r - rmin) * total) // span
                sr[pos[k]] = r
                sw[pos[k]] = wt
                pos[k] += 1
        # buckets are in order; finish with an insertion sort, which only has
        # to fix the order inside each bucket
        for k in range(1, total):
            r = sr[k]
            w = sw[k]
            m = k - 1
            while m >= 0 and sr[m] > r:
                sr[m + 1] = sr[m]
                sw[m + 1] = sw[m]
                m -= 1
            sr[m + 1] = r
            sw[m + 1] = w
        cum = 0.0
        qi = 0
        for k in range(total):
            cum += sw[k]
            while qi < nq and cum >= levels[qi] - 1e-12:
                out_q[i, qi] = ys[sr[k]]
                qi += 1
            if qi == nq:
                break
        while qi < nq:
            out_q[i, qi] = ys[sr[total - 1]]
            qi += 1


@njit(parallel=True, cache=True)
def _predict_one(Xb, ys, nodes, rep, levels, out_q, out_mean, block):
    """As ``_predict`` with one stored sample per leaf: every tree contributes
    one outcome with equal weight."""
    n = Xb.shape[0]
    T = nodes.shape[0]
    nq = levels.shape[0]
    for blk in prange((n + block - 1) // block):
        first = blk * block
        m = min(block, n - first)
        ranks = np.empty((m, T), dtype=np.int64)
        cur = np.empty(m, dtype=np.int64)
        act = np.empty(m, dtype=np.int64)
        for t in range(T):
            tree = nodes[t]
            cur[:] = 0
            for r in range(m):
                act[r] = r
            left = m
            while left > 0:  # only rows that have not reached a leaf yet
                k = 0
                for j in range(left):
                    r = act[j]
                    w = tree[cur[r]]
                    f = w >> 40
                    if f != 0:
                        cur[r] = (w & 0xFFFFFFFF) + (Xb[first + r, f - 1] > ((w >> 32) & 0xFF))
                        act[k] = r
                        k += 1
                left = k
            for r in range(m):
                ranks[r, t] = rep[t, cur[r]]
        for r in range(m):
            row = np.sort(ranks[r])
            mean = 0.0
            for t in range(T):
                mean += ys[row[t]]
            out_mean[first + r] = mean / T
            for qi in range(nq):
                k = int(np.ceil(levels[qi] * T - 1e-9)) - 1
                out_q[first + r, qi] = ys[row[min(max(k, 0), T - 1)]]


_NO_BAG = np.empty((0, 0), dtype=np.bool_)
_BLOCK = [0]  # 0 = choose from the number of rows and threads


def _block(n):
    """Rows handled together per thread. Larger blocks reuse each tree's nodes
    more before moving on; smaller blocks keep every thread busy."""
    if _BLOCK[0]:
        return _BLOCK[0]
    from numba import get_num_threads

    return int(min(4096, max(256, -(-n // (4 * get_num_threads())))))


class QuantileForest:
    """Quantile regression forest.

    Parameters
    ----------
    n_estimators : int, default=100
    min_samples_leaf : int, default=5
    max_features : float or int, default=1/3
        Features tried at each split: a fraction of all features, or a count.
    max_depth : int or None
    bootstrap : bool, default=True
        Draw each tree's sample with replacement.
    max_samples : float, int or None
        Size of each tree's sample (fraction or count). Default: all rows.
    max_bins : int, default=256
        Levels each feature is binned into (at most 256).
    max_samples_leaf : None or 1, default=None
        None keeps every training outcome of a leaf, which gives the exact
        quantile regression forest. 1 keeps one outcome per leaf, drawn at
        random: an approximation that predicts faster and is a little less
        accurate (the default of the quantile-forest package).
    random_state : int or None
        Results for a given seed do not depend on the number of threads.

    Use ``predict(X, quantiles=[...])`` for quantiles, shape
    (n_samples, n_levels) in the order given, or ``predict(X)`` for the
    conditional mean. A quantile is the smallest training outcome whose
    weighted share reaches the level, so level 0 gives the smallest and
    level 1 the largest outcome among the leaves reached.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 5,
        max_features=1.0 / 3.0,
        max_depth=None,
        bootstrap: bool = True,
        max_samples=None,
        max_bins: int = 256,
        random_state=None,
        max_samples_leaf=None,
    ):
        self.max_samples_leaf = max_samples_leaf
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_bins = max_bins
        self.random_state = random_state

    def get_params(self, deep: bool = False):
        return {k: getattr(self, k) for k in (
            "n_estimators", "min_samples_leaf", "max_features", "max_depth",
            "bootstrap", "max_samples", "max_bins", "random_state", "max_samples_leaf")}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                raise ValueError(f"Unknown parameter {k!r} for QuantileForest.")
            setattr(self, k, v)
        return self

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features).")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must be finite.")
        return X

    def _bin(self, X):
        Xb = np.empty(X.shape, dtype=np.uint8)
        for f, edges in enumerate(self.bin_edges_):
            Xb[:, f] = np.searchsorted(edges, X[:, f], side="left")
        return Xb

    def fit(self, X, y):
        if not HAVE_NUMBA:
            raise ImportError("QuantileForest needs numba: pip install pyconformal[fast]")
        X = self._matrix(X)
        y = np.ascontiguousarray(y, dtype=float).ravel()
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y have different numbers of rows.")
        if n == 0:
            raise ValueError("The training set is empty.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y must be finite.")
        if not 2 <= int(self.max_bins) <= 256:
            raise ValueError("max_bins must be between 2 and 256.")
        if int(self.min_samples_leaf) < 1 or int(self.n_estimators) < 1:
            raise ValueError("min_samples_leaf and n_estimators must be at least 1.")
        rng = np.random.default_rng(self.random_state)
        # bin edges: sample x goes to bin j when edges[j-1] < x <= edges[j]
        probs = np.linspace(0.0, 1.0, int(self.max_bins) + 1)[1:-1]
        sub = X if n <= 200_000 else X[rng.choice(n, 200_000, replace=False)]
        srt = np.sort(sub, axis=0)  # one sort gives every feature's quantiles
        pick = np.floor(probs * (srt.shape[0] - 1) + 1e-9).astype(np.int64)
        self.bin_edges_ = [np.unique(srt[pick, f]) for f in range(d)]
        Xb = self._bin(X)
        if self.max_samples is None:
            n_draw = n
        elif isinstance(self.max_samples, float):
            n_draw = max(1, int(round(self.max_samples * n)))
        else:
            n_draw = int(self.max_samples)
        if not 1 <= n_draw <= n and not self.bootstrap:
            raise ValueError("max_samples is out of range.")
        mf = self.max_features
        mtry = int(mf) if isinstance(mf, (int, np.integer)) else int(np.ceil(float(mf) * d))
        mtry = min(max(mtry, 1), d)
        min_leaf = int(self.min_samples_leaf)
        T = int(self.n_estimators)
        cap = 2 * max(n_draw // min_leaf, 1) + 1
        feat = np.empty((T, cap), dtype=np.int32)
        thr = np.empty((T, cap), dtype=np.uint8)
        a = np.empty((T, cap), dtype=np.int32)
        b = np.empty((T, cap), dtype=np.int32)
        idx = np.empty((T, n_draw), dtype=np.int32)
        n_nodes = np.empty(T, dtype=np.int64)
        seeds = rng.integers(1, 2**63 - 1, size=T).astype(np.uint64)
        depth = np.iinfo(np.int64).max if self.max_depth is None else int(self.max_depth)
        _build_forest(Xb, y, seeds, n_draw, bool(self.bootstrap), mtry, min_leaf, depth, feat, thr, a, b, idx, n_nodes)
        keep = int(n_nodes.max())
        feat, thr, a = feat[:, :keep], thr[:, :keep], a[:, :keep]
        self._nodes = (
            ((feat.astype(np.int64) + 1) << 40) | (thr.astype(np.int64) << 32) | a.astype(np.int64)
        )
        for t in range(T):  # unused slots must read as leaves
            self._nodes[t, n_nodes[t]:] = 0
        self._b = np.ascontiguousarray(b[:, :keep])
        # store each tree's samples as ranks among all outcomes
        order = np.argsort(y, kind="stable")
        rank = np.empty(n, dtype=np.int32)
        rank[order] = np.arange(n, dtype=np.int32)
        self._rk = rank[idx]
        self._order = order
        self._ys = y[order]
        if self.max_samples_leaf not in (None, 1):
            raise ValueError("max_samples_leaf must be None or 1.")
        self._rep = None
        if self.max_samples_leaf == 1:  # one outcome per leaf, chosen at random
            starts = self._nodes & 0xFFFFFFFF
            pick = starts + (rng.random(self._b.shape) * np.maximum(self._b, 1)).astype(np.int64)
            leaf = (self._nodes >> 40) == 0
            pick = np.where(leaf, np.minimum(pick, n_draw - 1), 0)
            self._rep = np.take_along_axis(self._rk, pick, axis=1)
        self.n_nodes_ = n_nodes
        self._Xb = Xb  # kept for out-of-bag predictions
        self.n_features_in_ = d
        return self

    def predict(self, X, quantiles=None):
        """Conditional mean (``quantiles=None``) or conditional quantiles."""
        if not hasattr(self, "_nodes"):
            raise RuntimeError("Call fit() first.")
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than in fit().")
        scalar = quantiles is not None and np.isscalar(quantiles)
        q = np.empty(0) if quantiles is None else np.atleast_1d(np.asarray(quantiles, dtype=float))
        if q.size and not (np.all(q >= 0.0) and np.all(q <= 1.0)):
            raise ValueError("Quantile levels must be between 0 and 1.")
        order = np.argsort(q, kind="stable")
        out_q = np.empty((X.shape[0], q.size))
        out_mean = np.empty(X.shape[0])
        if self._rep is not None:
            _predict_one(self._bin(X), self._ys, self._nodes, self._rep, q[order], out_q, out_mean, _block(X.shape[0]))
        else:
            _predict(self._bin(X), self._ys, self._nodes, self._b, self._rk, q[order], out_q, out_mean, _NO_BAG, _block(X.shape[0]))
        if quantiles is None:
            return out_mean
        res = np.empty_like(out_q)
        res[:, order] = out_q
        return res[:, 0] if scalar else res

    def predict_oob(self, quantiles=None):
        """Out-of-bag predictions for the training rows, in training order.

        Each row is predicted only by the trees whose sample did not contain
        it, so the result behaves like a prediction on unseen data and can
        be used to calibrate without setting data aside. Needs
        ``bootstrap=True`` or ``max_samples`` below the number of rows.
        """
        if not hasattr(self, "_nodes"):
            raise RuntimeError("Call fit() first.")
        if self._rep is not None:
            raise RuntimeError("Out-of-bag predictions need max_samples_leaf=None.")
        n = self._Xb.shape[0]
        T = self._nodes.shape[0]
        inbag = np.zeros((T, n), dtype=np.bool_)
        np.put_along_axis(inbag, self._order[self._rk], True, axis=1)
        scalar = quantiles is not None and np.isscalar(quantiles)
        q = np.empty(0) if quantiles is None else np.atleast_1d(np.asarray(quantiles, dtype=float))
        if q.size and not (np.all(q >= 0.0) and np.all(q <= 1.0)):
            raise ValueError("Quantile levels must be between 0 and 1.")
        order = np.argsort(q, kind="stable")
        out_q = np.empty((n, q.size))
        out_mean = np.empty(n)
        _predict(self._Xb, self._ys, self._nodes, self._b, self._rk, q[order], out_q, out_mean, inbag, _block(n))
        if np.any(np.isnan(out_mean)):
            raise RuntimeError(
                "Some rows were in every tree's sample, so they have no out-of-bag "
                "prediction. Use more trees, bootstrap=True or a smaller max_samples."
            )
        if quantiles is None:
            return out_mean
        res = np.empty_like(out_q)
        res[:, order] = out_q
        return res[:, 0] if scalar else res

