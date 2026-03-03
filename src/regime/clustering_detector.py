"""K-Means Clustering Regime Detector.

Unsupervised clustering on a richer feature set: [logret, rv_20, mom_20, z_ret_20].
States are sorted by ascending rv so labels are consistent with HMM / Vol detectors.
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import BaseRegimeDetector

logger = logging.getLogger(__name__)


class ClusteringRegimeDetector(BaseRegimeDetector):
    """K-Means clustering on [logret, rv_20, mom_20, z_ret_20].

    Parameters
    ----------
    n_states : int
        Number of clusters / regimes (default 3).
    feature_cols : list[str]
        Feature columns to use (must match featurize output).
    random_state : int
    """

    def __init__(
        self,
        n_states: int = 3,
        feature_cols: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.feature_cols = feature_cols or ["logret", "rv_20", "mom_20", "z_ret_20"]
        self.random_state = random_state
        self._model: Optional[KMeans] = None
        self._scaler = StandardScaler()
        self._label_map: dict = {}

    @property
    def n_regimes(self) -> int:
        return self.n_states

    def fit(self, df: pd.DataFrame) -> "ClusteringRegimeDetector":
        X, _ = self._prepare(df, fit_scaler=True)
        model = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=20)
        model.fit(X)
        self._model = model
        self._build_label_map()
        inertia = model.inertia_
        logger.info(f"KMeans fitted: {self.n_states} clusters, inertia={inertia:.2f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        X, idx = self._prepare(df, fit_scaler=False)
        raw = self._model.predict(X)
        mapped = np.array([self._label_map[s] for s in raw])
        return pd.Series(mapped, index=idx, name="regime", dtype=int)

    def state_stats(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        df2 = df.loc[labels.index].copy()
        df2["regime"] = labels
        rows = []
        for r in sorted(df2["regime"].unique()):
            sub = df2[df2["regime"] == r]
            row = {"regime": r, "n_days": len(sub)}
            for col in self.feature_cols:
                if col in sub.columns:
                    row[f"{col}_mean"] = sub[col].mean()
            if "logret" in sub.columns:
                row["ann_ret_pct"] = sub["logret"].mean() * 252 * 100
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")

    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame, fit_scaler: bool):
        avail = [c for c in self.feature_cols if c in df.columns]
        clean = df.dropna(subset=avail)
        X = clean[avail].values.astype(float)
        if fit_scaler:
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)
        return X, clean.index

    def _build_label_map(self):
        if self._model is None:
            return
        rv_idx = next(
            (i for i, c in enumerate(self.feature_cols) if "rv" in c), 0
        )
        centers = self._model.cluster_centers_[:, rv_idx]
        order = np.argsort(centers)
        self._label_map = {raw: sorted_idx for sorted_idx, raw in enumerate(order)}
