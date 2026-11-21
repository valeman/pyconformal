"""Finite-sample conformal quantile."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}.")
    return alpha


def conformal_quantile(scores, alpha: float, *, allow_inf: bool = False) -> float:
    """Return the split-conformal threshold for nonconformity ``scores``.

    The threshold is the ``k``-th smallest calibration score, where
    ``k = ceil((n + 1) * (1 - alpha))`` and ``n`` is the number of scores.
    If the calibration data and a test point are exchangeable, the test
    point's score is at most this threshold with probability at least
    ``1 - alpha``.

    If ``k > n`` (too few calibration points for the requested ``alpha``),
    the threshold is ``inf``: no finite threshold gives the guarantee.

    Parameters
    ----------
    scores : array-like of shape (n,)
        Nonconformity scores computed on the calibration set.
    alpha : float
        Miscoverage level, strictly between 0 and 1.

    Returns
    -------
    float
    """
    alpha = _check_alpha(alpha)
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        raise ValueError("scores must contain at least one value.")
    if allow_inf:  # scores of +inf are legitimate for some methods
        if np.any(np.isnan(scores)) or np.any(np.isneginf(scores)):
            raise ValueError("scores must not be NaN or -inf.")
    elif not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    # Exact rational arithmetic on the decimal value of alpha. Floating point
    # can land on the wrong side of a whole number: (49 + 1) * (1 - 0.58) is
    # 21.000000000000004, which a ceiling would round up to 22.
    k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def _band_py(y_hat, q):
    out = np.empty((2, y_hat.shape[0]))
    np.subtract(y_hat, q, out=out[0])
    np.add(y_hat, q, out=out[1])
    return out.T


try:  # pragma: no cover - exercised when numba is installed
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _band_nb(y_hat, q, out):
        for i in prange(y_hat.shape[0]):
            out[i, 0] = y_hat[i] - q
            out[i, 1] = y_hat[i] + q

    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

# Below this size the threads cost more than they save.
_PARALLEL_MIN = 100_000


def symmetric_band(y_hat, q):
    """Array of shape (n, 2) holding ``y_hat - q`` and ``y_hat + q``.

    Written in one pass; with numba installed, large arrays are filled in
    parallel. Both paths give identical values.
    """
    y_hat = np.ascontiguousarray(y_hat, dtype=np.float64)
    if _HAVE_NUMBA and y_hat.shape[0] >= _PARALLEL_MIN and math.isfinite(q):
        out = np.empty((y_hat.shape[0], 2))
        _band_nb(y_hat, float(q), out)
        return out
    return _band_py(y_hat, q)


def _scaled_py(y_hat, q, scale):
    out = np.empty((2, y_hat.shape[0]))
    half = q * scale
    np.subtract(y_hat, half, out=out[0])
    np.add(y_hat, half, out=out[1])
    return out.T


try:  # pragma: no cover
    from numba import njit as _njit, prange as _prange

    @_njit(parallel=True, cache=True)
    def _scaled_nb(y_hat, q, scale, out):
        for i in _prange(y_hat.shape[0]):
            h = q[i] * scale[i]
            out[i, 0] = y_hat[i] - h
            out[i, 1] = y_hat[i] + h

    _HAVE_SCALED = True
except Exception:
    _HAVE_SCALED = False


def scaled_band(y_hat, q, scale):
    """``y_hat -/+ q * scale`` as an (n, 2) array; ``q`` and ``scale`` are
    numbers or arrays of length n."""
    n = y_hat.shape[0]
    if _HAVE_SCALED and n >= _PARALLEL_MIN:
        out = np.empty((n, 2))
        _scaled_nb(y_hat, np.broadcast_to(np.asarray(q, dtype=float), (n,)),
                   np.broadcast_to(np.asarray(scale, dtype=float), (n,)), out)
        return out
    with np.errstate(invalid="ignore"):
        return _scaled_py(y_hat, q, scale)

