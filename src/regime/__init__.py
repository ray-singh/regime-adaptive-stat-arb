"""Regime detection modules."""

try:
    from .hmm_detector import HMMRegimeDetector
except ImportError:  # hmmlearn is optional
    HMMRegimeDetector = None  # type: ignore[assignment,misc]

from .volatility_detector import VolatilityRegimeDetector

try:
    from .clustering_detector import ClusteringRegimeDetector
except ImportError:  # scikit-learn is optional
    ClusteringRegimeDetector = None  # type: ignore[assignment,misc]

__all__ = ["HMMRegimeDetector", "VolatilityRegimeDetector", "ClusteringRegimeDetector"]
