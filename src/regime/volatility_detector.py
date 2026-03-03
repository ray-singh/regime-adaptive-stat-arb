"""Volatility-Threshold Regime Detector.

Fast, interpretable detector based on rolling realized volatility quantiles.
Uses rv_20 by default and splits into n_states buckets by percentile thresholds.

    n_states=2: low / high vol
    n_states=3: low / mid / high vol  (default)
    n_states=4: low / mid / high / crisis
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import logging

from .base import BaseRegimeDetector

logger = logging.getLogger(__name__)


class VolatilityRegimeDetector(BaseRegimeDetector):
    """Bucket regimes by quantile thresholds of realized volatility.

    Parameters
    ----------
    n_states : int
        Number of regimes (2–4).
    rv_col : str
        Column containing realized vol (default 'rv_20').
    """

    def __init__(self, n_states: int = 3, rv_col: str = "rv_20"):
        if not 2 <= n_states <= 4:
            raise ValueError("n_states must be 2, 3, or 4")
        self.n_states = n_states
        self.rv_col = rv_col
        self._thresholds: Optional[np.ndarray] = None

    @property
    def n_regimes(self) -> int:
        return self.n_states

    def fit(self, df: pd.DataFrame) -> "VolatilityRegimeDetector":
        if self.rv_col not in df.columns:
            raise KeyError(f"Column '{self.rv_col}' not found. Run featurize first.")
        rv = df[self.rv_col].dropna()
        qs = np.linspace(0, 1, self.n_states + 1)[1:-1]   # interior quantiles
        self._thresholds = rv.quantile(qs).values
        logger.info(f"VolatilityDetector fitted: thresholds={self._thresholds.round(4)}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._thresholds is None:
            raise RuntimeError("Call fit() first")
        rv = df[self.rv_col]
        labels = np.digitize(rv.values, self._thresholds).astype(int)
        return pd.Series(labels, index=df.index, name="regime")

    def state_stats(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        df2 = df.loc[labels.index].copy()
        df2["regime"] = labels
        rows = []
        for r in sorted(df2["regime"].unique()):
            sub = df2[df2["regime"] == r]
            row = {"regime": r, "n_days": len(sub),
                   "rv_mean": sub[self.rv_col].mean(),
                   "rv_p5": sub[self.rv_col].quantile(0.05),
                   "rv_p95": sub[self.rv_col].quantile(0.95)}
            if "logret" in sub.columns:
                row["ann_ret_pct"] = sub["logret"].mean() * 252 * 100
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")
