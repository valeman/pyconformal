"""Nonconformity scores for classification.

A score maps class probabilities to one number per class; lower means the
class conforms better. Every score here computes the scores of all classes
with one routine, and the score of the true label is read from that same
array. Calibration and prediction can therefore never disagree.

With ``pi_(1) >= pi_(2) >= ...`` the sorted probabilities of a sample, ``r``
the rank of a class (1 = most likely) and ``u`` a uniform random number
drawn once per sample:

* LAC:   ``1 - p``
* APS:   ``pi_(1) + ... + pi_(r-1) + u * pi_(r)``
* RAPS:  APS ``+ penalty * max(0, r - k_reg)``
* SAPS:  ``u * pi_(1)`` if ``r = 1``, else ``pi_(1) + (r - 2 + u) * weight``

With ``randomized=False``, ``u`` is fixed at 1. Coverage is at least
``1 - alpha`` either way; randomisation makes it (almost) exactly
``1 - alpha`` and the sets smaller.

References
----------
Sadinle, Lei and Wasserman (2019), LAC. Romano, Sesia and Candes (2020), APS.
Angelopoulos, Bates, Jordan and Malik (2021), RAPS. Huang, Xi, Zhang, Yao,
Qiu and Wei (2024), SAPS.
"""

from __future__ import annotations

import numpy as np

__all__ = ["LAC", "APS", "RAPS", "SAPS", "get_score"]


class _Score:
    randomized = False

    def scores(self, proba, u=None):
        """Scores of every class, shape (n_samples, n_classes).

        ``u`` holds one number in [0, 1] per sample; it is ignored by scores
        that are not randomised.
        """
        raise NotImplementedError

    def _u(self, n, u):
        if not self.randomized or u is None:
            return np.ones(n)
        u = np.asarray(u, dtype=float).ravel()
        if u.shape[0] != n:
            raise ValueError("u must hold one value per sample.")
        return u

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class LAC(_Score):
    """One minus the probability of the class. Smallest sets on average;
    sets can be empty."""

    def scores(self, proba, u=None):
        return 1.0 - np.asarray(proba, dtype=float)


def _sorted(proba):
    """Probabilities sorted in descending order, and the sorting permutation."""
    P = np.asarray(proba, dtype=float)
    order = np.argsort(P, axis=1)[:, ::-1]
    return np.take_along_axis(P, order, axis=1), order


def _unsort(ordered, order):
    """Put scores computed in sorted order back in class order."""
    out = np.empty_like(ordered)
    np.put_along_axis(out, order, ordered, axis=1)
    return out


class APS(_Score):
    """Adaptive prediction sets: cumulative probability down to the class.
    Sets adapt to how hard each sample is."""

    def __init__(self, randomized: bool = True):
        self.randomized = bool(randomized)

    def _ordered(self, srt, u):
        out = np.cumsum(srt, axis=1)
        if self.randomized:
            out -= (1.0 - u[:, None]) * srt
        return out

    def scores(self, proba, u=None):
        srt, order = _sorted(proba)
        return _unsort(self._ordered(srt, self._u(srt.shape[0], u)), order)


class RAPS(APS):
    """Regularised APS: adds ``penalty`` for every rank beyond ``k_reg``,
    which keeps unlikely classes out and makes sets smaller and more stable."""

    def __init__(self, penalty: float = 0.01, k_reg: int = 5, randomized: bool = True):
        if penalty < 0 or k_reg < 0:
            raise ValueError("penalty and k_reg must be >= 0.")
        self.penalty = float(penalty)
        self.k_reg = int(k_reg)
        self.randomized = bool(randomized)

    def _ordered(self, srt, u):
        r = np.arange(1, srt.shape[1] + 1)
        return super()._ordered(srt, u) + self.penalty * np.maximum(0, r - self.k_reg)[None, :]


class SAPS(_Score):
    """Sorted APS: uses only the top probability and the rank, so the score
    does not depend on the (often unreliable) tail probabilities."""

    def __init__(self, weight: float = 0.2, randomized: bool = True):
        if weight <= 0:
            raise ValueError("weight must be > 0.")
        self.weight = float(weight)
        self.randomized = bool(randomized)

    def scores(self, proba, u=None):
        srt, order = _sorted(proba)
        u = self._u(srt.shape[0], u)[:, None]
        top = srt[:, :1]
        ordered = top + (np.arange(-1, srt.shape[1] - 1)[None, :] + u) * self.weight
        ordered[:, 0] = (u * top)[:, 0]
        return _unsort(ordered, order)


_NAMES = {"lac": LAC, "aps": APS, "raps": RAPS, "saps": SAPS}


def get_score(score):
    """Turn a name ("lac", "aps", "raps", "saps") or a score object into a
    score object."""
    if isinstance(score, _Score):
        return score
    if isinstance(score, str) and score.lower() in _NAMES:
        return _NAMES[score.lower()]()
    raise ValueError(f"score must be one of {sorted(_NAMES)} or a score object, got {score!r}.")
