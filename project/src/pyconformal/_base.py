"""Shared plumbing: every method works from model outputs, or from a model.

* Without an estimator, methods take the model's outputs by keyword
  (``y_pred=``, ``proba=`` or ``scores=``).
* With an estimator, methods take ``X`` and call the model themselves.

Either way the same code does the conformal part.
"""

from __future__ import annotations

import inspect

import numpy as np

from pyconformal._quantile import _check_alpha


def _rows(fit_params, idx, n):
    """Fit parameters for a subset or resample of the rows: arrays with one
    entry per training row (such as ``sample_weight``) follow ``idx``;
    everything else is passed unchanged."""
    out = {}
    for k, v in fit_params.items():
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes, dict)):
            a = np.asarray(v)
            if a.ndim >= 1 and a.shape[0] == n:
                out[k] = a[idx]
                continue
        out[k] = v
    return out


class _Conformal:
    _model_method = "predict"  # what to call on the estimator
    _given = "y_pred"  # keyword that carries model outputs
    # attributes that hold the calibration; cleared when the model is refitted
    _calibration_attrs: tuple = ("calibration_scores_",)

    estimator = None

    def _forget_calibration(self):
        for a in self._calibration_attrs:
            if hasattr(self, a):
                setattr(self, a, None)

    def fit(self, X, y, **fit_params):
        """Train the underlying model on the proper training set. Skip this
        when the model is already trained."""
        if self.estimator is None:
            raise TypeError(
                "fit() trains the underlying model, and no estimator was given. "
                "With model outputs, go straight to calibrate()."
            )
        self.estimator.fit(X, y, **fit_params)
        # a new model makes the old calibration invalid: calibrate again
        self._forget_calibration()
        return self

    def _output(self, X, given):
        """Model outputs, from the keyword or by calling the estimator."""
        name = self._given
        if given is not None:
            if X is not None:
                raise TypeError(f"Pass X or {name}=, not both.")
            return np.asarray(given, dtype=float)
        if X is None:
            raise TypeError(f"Pass X (with an estimator) or {name}= (model outputs).")
        if self.estimator is None:
            raise TypeError(
                "No estimator was given, so the first argument cannot be X. "
                f"Pass the model's outputs by keyword: {name}=..."
            )
        return np.asarray(getattr(self.estimator, self._model_method)(X), dtype=float)

    def _alphas(self, alpha):
        """(list of levels, whether a single number was asked for)."""
        if alpha is None:
            return [self.alpha], True
        if np.isscalar(alpha):
            return [_check_alpha(alpha)], True
        return [_check_alpha(a) for a in alpha], False

    # scikit-learn conventions, without depending on scikit-learn
    def get_params(self, deep: bool = False):
        names = [p for p in inspect.signature(type(self).__init__).parameters if p != "self"]
        return {n: getattr(self, n) for n in names}

    def set_params(self, **params):
        """Change parameters. They are checked by building a new object
        first, so a bad value leaves this one unchanged. Changing only
        ``alpha`` keeps the calibration; any other change clears it."""
        valid = self.get_params()
        for k in params:
            if k not in valid:
                raise ValueError(f"Unknown parameter {k!r} for {type(self).__name__}.")
        fresh = type(self)(**{**valid, **params})
        if set(params) <= {"alpha"}:
            self.alpha = fresh.alpha
            return self
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self
