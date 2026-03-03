"""Gaussian HMM Regime Detector.

Fits a Gaussian HMM on daily log-returns and realized volatility,
then relabels states so regime 0 = lowest vol (bull), n-1 = highest vol (crisis).
"""

import numpy as np
import pandas as pd
from typing import List, Optional
import logging
import warnings

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .base import BaseRegimeDetector

logger = logging.getLogger(__name__)


class HMMRegimeDetector(BaseRegimeDetector):
    """Gaussian HMM on [logret, rv_20] (or user-chosen features).

    States are automatically sorted by ascending realized volatility so:
        0 = Bull / Low-Vol
        1 = Neutral / Mid-Vol
        2 = Bear / High-Vol
        3 = Crisis / Extreme-Vol (if n_states=4)

    Parameters
    ----------
    n_states : int
        Number of hidden states (default 3).
    feature_cols : list[str]
        Columns used for fitting (must exist after featurize). Default uses
        logret + rv_20. Pass extra cols like mom_20 for richer models.
    n_iter : int
        Max EM iterations.
    random_state : int
    """

    def __init__(
        self,
        n_states: int = 3,
        feature_cols: Optional[List[str]] = None,
        n_iter: int = 1000,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.feature_cols = feature_cols or ["logret", "rv_20"]
        self.n_iter = n_iter
        self.random_state = random_state

        self._model: Optional[GaussianHMM] = None
        self._scaler = StandardScaler()
        self._label_map: dict = {}   # raw HMM state -> sorted state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit on a single-ticker feature DataFrame (output of featurize)."""
        X = self._prepare(df, fit_scaler=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False,
            )
            model.fit(X)
        self._model = model
        self._build_label_map()
        logger.info(
            f"HMM fitted: {self.n_states} states, converged={model.monitor_.converged}, "
            f"score={model.score(X):.2f}"
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels aligned to df's DatetimeIndex."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        X = self._prepare(df, fit_scaler=False)
        raw_states = self._model.predict(X)
        mapped = np.array([self._label_map[s] for s in raw_states])
        idx = df.dropna(subset=self.feature_cols).index
        return pd.Series(mapped, index=idx, name="regime", dtype=int)

    @property
    def n_regimes(self) -> int:
        return self.n_states

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def transition_matrix(self) -> pd.DataFrame:
        """Return the HMM transition matrix with sorted labels."""
        if self._model is None:
            raise RuntimeError("Call fit() first")
        raw = self._model.transmat_
        n = self.n_states
        # re-order rows/cols by label map
        order = [k for k, v in sorted(self._label_map.items(), key=lambda x: x[1])]
        reordered = raw[np.ix_(order, order)]
        return pd.DataFrame(reordered, index=range(n), columns=range(n)).round(4)

    def state_stats(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        """Return mean/std of each feature per regime."""
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        df2 = df.loc[labels.index].copy()
        df2["regime"] = labels

        rows = []
        for r in sorted(df2["regime"].unique()):
            sub = df2[df2["regime"] == r]
            row = {"regime": r, "n_days": len(sub)}
            for col in self.feature_cols:
                if col in sub.columns:
                    row[f"{col}_mean"] = sub[col].mean()
                    row[f"{col}_std"] = sub[col].std()
            # annualized return
            if "logret" in sub.columns:
                row["ann_return_pct"] = sub["logret"].mean() * 252 * 100
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")

    def score(self, df: pd.DataFrame) -> float:
        """Log-likelihood per sample."""
        if self._model is None:
            raise RuntimeError("Call fit() first")
        X = self._prepare(df, fit_scaler=False)
        return self._model.score(X) / len(X)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame, fit_scaler: bool) -> np.ndarray:
        clean = df.dropna(subset=self.feature_cols)
        X = clean[self.feature_cols].values.astype(float)
        if fit_scaler:
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)
        return X

    def _build_label_map(self):
        """Map raw HMM states to sorted labels (0=lowest vol)."""
        if self._model is None:
            return
        # Use the rv (first or second col containing "rv") mean to rank
        rv_idx = next(
            (i for i, c in enumerate(self.feature_cols) if "rv" in c),
            0   # fall back to first col
        )
        # Get scaler-unscaled means
        means_scaled = self._model.means_[:, rv_idx]
        order = np.argsort(means_scaled)   # ascending vol -> state 0 = lowest
        self._label_map = {raw: sorted_idx for sorted_idx, raw in enumerate(order)}
