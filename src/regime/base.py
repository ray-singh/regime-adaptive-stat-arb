"""Abstract base class for regime detectors."""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional


# Canonical regime labels (low index = benign)
REGIME_LABELS = {
    0: "Bull / Low-Vol",
    1: "Neutral / Mid-Vol",
    2: "Bear / High-Vol",
    3: "Crisis / Extreme-Vol",
}


class BaseRegimeDetector(ABC):
    """All regime detectors must implement fit() and predict()."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRegimeDetector":
        """Fit the detector on a single-ticker feature DataFrame."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return integer regime labels aligned to df's index."""

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        self.fit(df)
        return self.predict(df)

    @property
    @abstractmethod
    def n_regimes(self) -> int:
        """Number of distinct regimes."""

    def regime_summary(self, labels: pd.Series) -> pd.DataFrame:
        """Count and pct of observations per regime."""
        counts = labels.value_counts().sort_index()
        pct = (counts / len(labels) * 100).round(1)
        return pd.DataFrame({"count": counts, "pct": pct,
                             "label": [REGIME_LABELS.get(i, f"Regime {i}") for i in counts.index]})
