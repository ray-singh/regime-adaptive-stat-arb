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
        # mom_20 is added so the HMM can distinguish trending from mean-reverting markets (spec §3.3)
        self.feature_cols = feature_cols or ["logret", "rv_20", "mom_20"]
        self.n_iter = n_iter
        self.random_state = random_state

        self._model: Optional[GaussianHMM] = None
        self._scaler = StandardScaler()
        self._label_map: dict = {}   # raw HMM state -> sorted state
        self._active_features: list = []  # columns actually used during fit
        # Walk-forward labels: pre-computed bias-free regime series (guide §3)
        self._walkforward_labels: Optional[pd.Series] = None

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

    def fit_predict_walkforward(
        self,
        df: pd.DataFrame,
        min_train_years: int = 2,
        retrain_every_years: int = 1,
    ) -> pd.Series:
        """Generate look-ahead-free regime labels via walk-forward training (guide §3).

        Walk-forward scheme:
          - Initial fit: first `min_train_years` * 252 bars.
          - Then for each subsequent `retrain_every_years`-year chunk:
              * Fit HMM on all prior bars (expanding window).
              * Predict labels for that chunk only.
          This prevents any future data leaking into past regime labels.

        At the end a fallback model is fit using only basic features
        (logret, rv_20, mom_20) so that online inference in the backtest
        (which has no macro/volume data) still works safely.

        Stores computed labels in ``self._walkforward_labels`` and returns them.
        """
        BASIC_COLS = ["logret", "rv_20", "mom_20"]
        bars_per_year  = 252
        min_train_bars = min_train_years * bars_per_year
        retrain_every  = retrain_every_years * bars_per_year

        # All feature cols present in df
        wf_cols = [c for c in self.feature_cols if c in df.columns]
        if not wf_cols:
            raise ValueError(
                f"None of the required feature columns {self.feature_cols} found. "
                f"Available: {list(df.columns)}"
            )

        clean = df.dropna(subset=wf_cols).copy()
        n = len(clean)

        if n <= min_train_bars:
            logger.warning(
                "Insufficient data for walk-forward (%d bars < %d required). "
                "Falling back to single-split fit.", n, min_train_bars
            )
            self.fit(df)
            labels = self.predict(df)
            self._walkforward_labels = labels
            return labels

        # ------- Walk-forward loop -------
        all_labels: dict = {}   # date -> int regime
        starts = list(range(min_train_bars, n, retrain_every))

        for chunk_start in starts:
            chunk_end   = min(chunk_start + retrain_every, n)
            train_chunk = clean.iloc[:chunk_start]
            test_chunk  = clean.iloc[chunk_start:chunk_end]

            X_train = train_chunk[wf_cols].values.astype(float)
            scaler  = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_sc)

            # Sort states by ascending realized vol (0 = bull, n-1 = crisis)
            rv_idx   = next((i for i, c in enumerate(wf_cols) if "rv" in c), 0)
            order    = np.argsort(model.means_[:, rv_idx])
            label_map = {raw: idx for idx, raw in enumerate(order)}

            X_test   = test_chunk[wf_cols].values.astype(float)
            X_test_sc = scaler.transform(X_test)
            raw_states = model.predict(X_test_sc)
            mapped   = [label_map[s] for s in raw_states]

            for date, lbl in zip(test_chunk.index, mapped):
                all_labels[date] = int(lbl)

            logger.info(
                "Walk-forward chunk: train=%d bars (up to %s), predict %d bars [%s → %s]",
                chunk_start,
                clean.index[chunk_start - 1],
                len(test_chunk),
                clean.index[chunk_start],
                clean.index[chunk_end - 1],
            )

        # ------- Fit fallback model on BASIC cols only (safe for online inference) -------
        basic_avail = [c for c in BASIC_COLS if c in df.columns]
        if not basic_avail:
            basic_avail = wf_cols  # last resort
        self._active_features = basic_avail
        basic_clean = df.dropna(subset=basic_avail)
        X_basic = basic_clean[basic_avail].values.astype(float)
        self._scaler.fit(X_basic)
        X_sc = self._scaler.transform(X_basic)
        fallback = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fallback.fit(X_sc)
        self._model = fallback
        self._build_label_map()

        labels_series = pd.Series(all_labels, name="regime", dtype=int)
        labels_series.index = pd.DatetimeIndex(labels_series.index)
        self._walkforward_labels = labels_series
        logger.info(
            "Walk-forward complete: %d regime labels generated over %d total bars, features=%s",
            len(labels_series), n, wf_cols,
        )
        return labels_series

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels aligned to df's DatetimeIndex.

        Prefers pre-computed walk-forward labels (bias-free) when available.
        Falls back to online model-based prediction for any uncovered dates.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")

        active = self._active_features or [c for c in self.feature_cols if c in df.columns]
        avail  = [c for c in active if c in df.columns]
        if not avail:
            avail = [c for c in self.feature_cols if c in df.columns]
        clean_idx = df.dropna(subset=avail).index if avail else df.index

        # --- Walk-forward labels: prefer them to avoid look-ahead bias ---
        if self._walkforward_labels is not None and len(clean_idx) > 0:
            wf_aligned = self._walkforward_labels.reindex(clean_idx, method="ffill")
            wf_aligned = wf_aligned.bfill()  # fill start (before first WF prediction)
            if wf_aligned.notna().all():
                return wf_aligned.astype(int).rename("regime")

        # --- Online (model-based) fallback ---
        X          = self._prepare(df, fit_scaler=False)
        raw_states = self._model.predict(X)
        mapped     = np.array([self._label_map[s] for s in raw_states])
        idx        = df.dropna(subset=[c for c in (self._active_features or self.feature_cols) if c in df.columns]).index
        online     = pd.Series(mapped, index=idx, name="regime", dtype=int)

        # Blend: use WF where available, online elsewhere
        if self._walkforward_labels is not None:
            common = online.index.intersection(self._walkforward_labels.index)
            online.loc[common] = self._walkforward_labels.loc[common]
        return online

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
        return self._model.score(X) / max(len(X), 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame, fit_scaler: bool) -> np.ndarray:
        # Use only columns that are present in this df (graceful fallback)
        avail = [c for c in self.feature_cols if c in df.columns]
        if not avail:
            raise ValueError(
                f"None of the required feature columns {self.feature_cols} found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        if fit_scaler:
            self._active_features = avail  # remember exactly which cols were used at fit time
        else:
            # Use the same columns that were seen during fit
            avail = self._active_features if self._active_features else avail
        clean = df.dropna(subset=avail)
        X = clean[avail].values.astype(float)
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
