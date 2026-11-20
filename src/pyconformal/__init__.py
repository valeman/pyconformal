"""pyconformal: conformal prediction for Python."""

from pyconformal._quantile import conformal_quantile
from pyconformal.classification import SplitConformalClassifier
from pyconformal.csp import ConformalSeasonalPool, CSPResult
from pyconformal.online import ACIResult, AdaptiveConformalInference
from pyconformal.predictive import ConformalPredictiveSystem
from pyconformal.regression import SplitConformalRegressor
from pyconformal.scores import APS, LAC, RAPS, SAPS
from pyconformal.venn_abers import MulticlassVennAbers, VennAbers, VennAbersCalibrator

__version__ = "0.0.1"

__all__ = [
    "APS",
    "LAC",
    "RAPS",
    "SAPS",
    "ACIResult",
    "AdaptiveConformalInference",
    "CSPResult",
    "ConformalPredictiveSystem",
    "ConformalSeasonalPool",
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "MulticlassVennAbers",
    "VennAbers",
    "VennAbersCalibrator",
    "conformal_quantile",
    "__version__",
]
