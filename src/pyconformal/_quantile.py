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


def conformal_quantile(scores, alpha: float) -> float:
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
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite.")
    # Exact rational arithmetic: in floating point (19 + 1) * (1 - 0.05)
    # is 19.000000000000004, which would wrongly round k up to 20.
    k = math.ceil((n + 1) * (1 - Fraction(repr(alpha))))
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])
