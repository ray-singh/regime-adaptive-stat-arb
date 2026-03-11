"""Simple Feature Store for computed features.

Stores per-ticker feature Parquet files and provides batch computation helpers.
"""

from pathlib import Path
import pandas as pd
from typing import Optional, List
from .featurize import compute_standard_features
import logging

logger = logging.getLogger(__name__)


class FeatureStore:
    """Feature store that caches computed features per ticker as Parquet.

    Usage:
        store = FeatureStore(cache_dir='data/features')
        store.compute_and_store(ticker, df)
        df_feats = store.load(ticker)
    """

    def __init__(self, cache_dir: Optional[str] = 'data/features'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_features.parquet"

    def compute_and_store(self, ticker: str, raw_df: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """Compute standard features for `raw_df` and store them on disk.

        Returns computed DataFrame.
        """
        windows = windows or [5, 20, 60]
        df_feats = compute_standard_features(raw_df, windows=windows)
        path = self._path_for(ticker)
        try:
            df_feats.to_parquet(path, index=False, compression='snappy')
            logger.info(f"Saved features for {ticker} to {path}")
        except Exception as e:
            logger.warning(f"Failed to save features for {ticker}: {e}")
        return df_feats

    def load(self, ticker: str) -> pd.DataFrame:
        """Load features for a ticker from cache; empty DataFrame if missing."""
        path = self._path_for(ticker)
        if not path.exists():
            logger.info(f"Feature file not found for {ticker}: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            logger.warning(f"Failed to load features for {ticker}: {e}")
            return pd.DataFrame()

    def batch_compute(self, tickers: List[str], client, period: str = '10y', interval: str = '1d', windows: Optional[List[int]] = None, show_progress: bool = True):
        """Fetch raw data for each ticker via `client` (must implement `fetch_ticker`) and compute/store features.

        Returns list of tickers successfully processed.
        """
        succeeded = []
        total = len(tickers)
        for i, t in enumerate(tickers, 1):
            if show_progress and i % 10 == 0:
                logger.info(f"FeatureStore progress: {i}/{total}")
            raw = client.fetch_ticker(t, period=period, interval=interval)
            if raw.empty:
                logger.warning(f"No raw data for {t}; skipping feature compute")
                continue
            try:
                self.compute_and_store(t, raw, windows=windows)
                succeeded.append(t)
            except Exception as e:
                logger.warning(f"Failed to compute features for {t}: {e}")
        return succeeded

    def compute_and_store_market_features(self, wide_df: "pd.DataFrame", window: int = 20, filename: str = "market_features") -> "pd.DataFrame":
        """Compute cross-asset market-level features and persist them as the shared feature table.

        This is the spec §2 'feature table' — one row per date with market-wide
        signals (ret_dispersion, avg_rv, avg_corr, mom_dispersion, index_momentum)
        used as inputs to the Regime Detection module.

        Parameters
        ----------
        wide_df : DataFrame
            Wide price matrix (Date × Ticker) — typically from
            ``YFinanceClient.get_price_matrix()``.
        window : int
            Rolling window for volatility / correlation estimates.
        filename : str
            Parquet filename stem (without extension) stored under cache_dir.

        Returns
        -------
        DataFrame with DatetimeIndex and market-level feature columns.
        """
        from .featurize import compute_market_features
        features = compute_market_features(wide_df, window=window)
        path = self.cache_dir / f"{filename}.parquet"
        try:
            features.to_parquet(path, compression="snappy")
            logger.info(f"Saved market features ({len(features)} rows) to {path}")
        except Exception as e:
            logger.warning(f"Failed to save market features: {e}")
        return features

    def load_market_features(self, filename: str = "market_features") -> "pd.DataFrame":
        """Load the persisted market feature table; returns empty DataFrame if not found."""
        path = self.cache_dir / f"{filename}.parquet"
        if not path.exists():
            logger.info(f"Market feature file not found: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.warning(f"Failed to load market features: {e}")
            return pd.DataFrame()
