"""Periodic pair re-selection.

Addresses the key failure mode of static pair selection: pairs that are
cointegrated in-sample may lose cointegration out-of-sample.  This module
periodically re-runs pair selection on a trailing price window and dynamically
updates the strategy's pair set.

Key concepts:
    - `reselection_interval_days`: how often to re-select (e.g. every 63 trading days)
    - `lookback_days`: how much trailing price history to use for the cointegration test
    - Pairs that are no longer cointegrated are closed out
    - Newly cointegrated pairs are added (up to max_open_pairs)
    - Hedge ratios are re-estimated on each re-selection

Integration:
    PairReSelector is called from PairsBacktestStrategy.on_market_event()
    when the bar count hits the next reselection epoch.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd

from strategy.pairs_trading import PairsSelector

logger = logging.getLogger(__name__)


class PairReSelector:
    """Periodically re-select cointegrated pairs on trailing data.

    Parameters
    ----------
    reselection_interval : int
        Number of trading days between re-selections.
    lookback_days : int
        Trailing window of prices for cointegration test.
    pvalue_threshold : float
        Max p-value for Engle-Granger test.
    min_half_life : int
        Min mean-reversion half-life.
    max_half_life : int
        Max mean-reversion half-life.
    max_pairs : int
        Maximum number of pairs to keep.
    """

    def __init__(
        self,
        reselection_interval: int = 63,
        lookback_days: int = 504,
        pvalue_threshold: float = 0.05,
        min_half_life: int = 5,
        max_half_life: int = 126,
        max_pairs: int = 10,
    ):
        self.reselection_interval = reselection_interval
        self.lookback_days = lookback_days
        self.max_pairs = max_pairs
        self._selector = PairsSelector(
            pvalue_threshold=pvalue_threshold,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
        )
        self._last_reselection_bar: int = 0
        self._reselection_count: int = 0

    def should_reselect(self, bar_count: int) -> bool:
        """Return True if it's time to re-select pairs."""
        if bar_count - self._last_reselection_bar >= self.reselection_interval:
            return True
        return False

    def reselect(
        self,
        bar_count: int,
        price_history: pd.DataFrame,
        current_pair_ids: set[str],
    ) -> tuple[pd.DataFrame, set[str], set[str]]:
        """Re-run pair selection on the trailing price window.

        Parameters
        ----------
        bar_count : int
            Current bar index.
        price_history : DataFrame
            Wide price matrix with enough trailing history.
        current_pair_ids : set[str]
            Set of currently active pair IDs (e.g. {"AAPL/AMD", "MSFT/GOOGL"}).

        Returns
        -------
        (new_pairs_df, added_pair_ids, removed_pair_ids)
            new_pairs_df: DataFrame of pairs from find_pairs()
            added_pair_ids: pairs to add
            removed_pair_ids: pairs to close out
        """
        self._last_reselection_bar = bar_count
        self._reselection_count += 1

        # Use trailing lookback window
        if len(price_history) > self.lookback_days:
            window = price_history.iloc[-self.lookback_days:]
        else:
            window = price_history

        # Drop tickers with too many NaNs
        valid_cols = window.columns[window.notna().sum() > 252]
        window = window[valid_cols].dropna(axis=0, how="all")

        if window.shape[1] < 2 or len(window) < 252:
            logger.warning("Insufficient data for pair re-selection (shape=%s)", window.shape)
            return pd.DataFrame(), set(), set()

        new_pairs_df = self._selector.find_pairs(
            window, max_pairs=self.max_pairs, verbose=False
        )

        if new_pairs_df.empty:
            logger.info("Pair re-selection #%d at bar %d: no pairs found, keeping current",
                        self._reselection_count, bar_count)
            return new_pairs_df, set(), set()

        # Vectorized set construction — avoids per-row Python overhead of iterrows
        new_pair_ids = set(new_pairs_df["ticker1"] + "/" + new_pairs_df["ticker2"])

        added = new_pair_ids - current_pair_ids
        removed = current_pair_ids - new_pair_ids

        logger.info(
            "Pair re-selection #%d at bar %d: kept=%d, added=%d, removed=%d",
            self._reselection_count, bar_count,
            len(new_pair_ids & current_pair_ids), len(added), len(removed),
        )

        return new_pairs_df, added, removed

    @property
    def reselection_count(self) -> int:
        return self._reselection_count
