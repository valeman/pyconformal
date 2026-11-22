"""Quantile predictions from CatBoost, XGBoost or LightGBM, in the form the
quantile-based conformal methods expect.

CQR needs two quantile levels; CTI and CIR need many, from one model, through
``predict(X, quantiles=[...])``. The three boosting libraries each train
quantile models differently:

* CatBoost: one model for all levels (``MultiQuantile`` loss);
* XGBoost: one model for all levels (``reg:quantileerror`` with a list of
  levels; needs XGBoost 2.0 or later);
* LightGBM: one model per level.

:class:`QuantileBooster` hides the difference. None of the libraries is a
requirement of pyconformal: install the one you use.
"""

from __future__ import annotations

import numpy as np

from pyconformal._base import _rows

__all__ = ["QuantileBooster"]

_LIBRARIES = ("catboost", "xgboost", "lightgbm")


class QuantileBooster:
    """Gradient-boosted quantile regression with CatBoost, XGBoost or LightGBM.

    ::

        qb = QuantileBooster("catboost", levels=19, iterations=300)
        cir = ConformalInterquantileRegressor(qb, alpha=0.1)
        cir.fit(X_train, y_train).calibrate(X_cal, y_cal)
        cir.predict_interval(X_test)

    Parameters
    ----------
    library : {"catboost", "xgboost", "lightgbm"}
    levels : int or array-like, default=19
        Quantile levels to train. A number ``k`` means ``k`` equally spaced
        levels strictly between 0 and 1 (19 gives 0.05, 0.10, ..., 0.95).
        Training cost grows with the number of levels, most of all for
        LightGBM, which trains one model per level.
    tail_holdout : float, default=0.1
        Share of the training rows kept out of the boosting and used only to
        set the tails (levels 0 and 1, see below). Boosted models fit their
        training rows closely, so tails measured on those rows alone are too
        short, and CIR and CTI can then return infinite intervals. 0 uses
        the training rows themselves.
    **params
        Passed to the library's regressor (``iterations``, ``n_estimators``,
        ``learning_rate``, ``max_depth``, ...). Output is silenced unless
        you set the library's own verbosity parameter.

    ``predict(X, quantiles=q)`` returns one column per requested level:

    * the trained quantiles are sorted within each row first, so they never
      cross;
    * a level between two trained levels is interpolated linearly;
    * levels 0 and 1 are the lowest and highest trained quantiles moved out
      by a fixed distance (``tails_``), measured on the held-out rows so
      that about 1% of new outcomes fall beyond each end. (With fewer than
      100 held-out rows the distance is the largest one over all rows passed
      to ``fit``.) For ``alpha`` below about 0.05, CIR and CTI may then
      need the infinite tails; train more extreme levels for that. Levels
      between a tail and the nearest trained level are interpolated. Boosted
      models cannot estimate a minimum or maximum, so treat these columns as
      a rough tail.

    The conformal step that follows guarantees coverage whatever the quality
    of these quantiles; better quantiles give shorter intervals.

    Attributes
    ----------
    levels_ : ndarray
        Trained levels, increasing.
    models_ : list
        The fitted library models (one, or one per level for LightGBM).
    tails_ : (float, float)
        Distances from the lowest and highest trained quantiles to levels 0
        and 1.
    """

    def __init__(self, library: str = "catboost", levels=19, tail_holdout: float = 0.1, **params):
        if library not in _LIBRARIES:
            raise ValueError(f"library must be one of {_LIBRARIES}, got {library!r}.")
        self.library = library
        self.levels = levels
        if not 0.0 <= float(tail_holdout) < 1.0:
            raise ValueError("tail_holdout must be at least 0 and below 1.")
        self.tail_holdout = float(tail_holdout)
        self.params = dict(params)
        self.levels_ = self._check_levels(levels)
        self.models_ = None

    @staticmethod
    def _check_levels(levels):
        if np.isscalar(levels):
            k = int(levels)
            if k != levels or k < 2:
                raise ValueError("levels must be an integer of at least 2, or an array of levels.")
            return np.arange(1, k + 1) / (k + 1.0)
        lv = np.unique(np.asarray(levels, dtype=float).ravel())
        if lv.size < 2 or not (lv[0] > 0.0 and lv[-1] < 1.0):
            raise ValueError("Give at least two distinct levels strictly between 0 and 1.")
        return lv

    def get_params(self, deep: bool = False):
        return {"library": self.library, "levels": self.levels, "tail_holdout": self.tail_holdout, **self.params}

    def set_params(self, **params):
        new = {**self.get_params(), **params}
        fresh = type(self)(**new)
        self.__dict__.update(fresh.__dict__)
        return self

    # ------------------------------------------------------------- training

    def _build(self):
        lv = [float(v) for v in self.levels_]
        p = dict(self.params)
        if self.library == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError as err:  # pragma: no cover
                raise ImportError("QuantileBooster('catboost') needs catboost: pip install catboost") from err
            if "verbose" not in p and "logging_level" not in p and "silent" not in p:
                p["verbose"] = 0
            p.setdefault("allow_writing_files", False)
            alphas = ",".join(repr(v) for v in lv)
            return [CatBoostRegressor(loss_function=f"MultiQuantile:alpha={alphas}", **p)]
        if self.library == "xgboost":
            try:
                import xgboost
            except ImportError as err:  # pragma: no cover
                raise ImportError("QuantileBooster('xgboost') needs xgboost: pip install xgboost") from err
            if int(xgboost.__version__.split(".")[0]) < 2:
                raise ImportError("QuantileBooster('xgboost') needs XGBoost 2.0 or later.")
            p.setdefault("verbosity", 0)
            return [xgboost.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.asarray(lv), **p)]
        try:
            from lightgbm import LGBMRegressor
        except ImportError as err:  # pragma: no cover
            raise ImportError("QuantileBooster('lightgbm') needs lightgbm: pip install lightgbm") from err
        p.setdefault("verbose", -1)
        return [LGBMRegressor(objective="quantile", alpha=v, **p) for v in lv]

    def fit(self, X, y, **fit_params):
        """Train on the proper training set. ``fit_params`` (for example
        ``sample_weight``) go to the library's ``fit``."""
        y = np.asarray(y, dtype=float).ravel()
        n = y.shape[0]
        held = int(round(self.tail_holdout * n))
        models = self._build()
        if held >= 1 and n - held >= 2:
            keep = np.ones(n, dtype=bool)
            keep[np.random.default_rng(0).choice(n, size=held, replace=False)] = False
            Xa = X if hasattr(X, "iloc") else np.asarray(X)
            Xk = Xa.iloc[keep] if hasattr(Xa, "iloc") else Xa[keep]
            for m in models:
                m.fit(Xk, y[keep], **_rows(fit_params, keep, n))
        else:
            for m in models:
                m.fit(X, y, **fit_params)
        self.models_ = models
        Q = self._trained(X)
        below, above = np.maximum(Q[:, 0] - y, 0.0), np.maximum(y - Q[:, -1], 0.0)
        if held >= 100:  # enough unseen rows: all but about 1% of them, at each end
            out = ~keep
            self.tails_ = (float(np.quantile(below[out], 0.99)), float(np.quantile(above[out], 0.99)))
        else:  # few rows: every outcome passed to fit
            self.tails_ = (float(below.max()), float(above.max()))
        return self

    # ----------------------------------------------------------- prediction

    def _trained(self, X):
        """Trained quantiles, shape (n_samples, n_levels), sorted in each row."""
        if self.models_ is None:
            raise RuntimeError("Call fit() first.")
        k = self.levels_.size
        if len(self.models_) == 1:
            Q = np.asarray(self.models_[0].predict(X), dtype=float).reshape(-1, k)
        else:
            Q = np.column_stack([np.asarray(m.predict(X), dtype=float).ravel() for m in self.models_])
        Q.sort(axis=1)
        return Q

    def predict(self, X, quantiles=None):
        """The median (``quantiles=None``) or the requested quantiles, shape
        (n_samples, n_quantiles); (n_samples,) for a single number."""
        scalar = quantiles is None or np.isscalar(quantiles)
        q = np.atleast_1d(np.asarray(0.5 if quantiles is None else quantiles, dtype=float))
        if not (np.all(q >= 0.0) and np.all(q <= 1.0)):
            raise ValueError("Quantile levels must be between 0 and 1.")
        Q = self._trained(X)
        # trained grid extended by the two tails at levels 0 and 1
        L = np.concatenate([[0.0], self.levels_, [1.0]])
        Q = np.column_stack([Q[:, 0] - self.tails_[0], Q, Q[:, -1] + self.tails_[1]])
        j = np.clip(np.searchsorted(L, q, side="right") - 1, 0, L.size - 2)
        w = (q - L[j]) / (L[j + 1] - L[j])
        out = Q[:, j] + w[None, :] * (Q[:, j + 1] - Q[:, j])
        return out[:, 0] if scalar else out
