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
from utils.pair_id import make_pair_id, split_pair_id

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
        # Default cadence
        effective_interval = self.reselection_interval
        return (bar_count - self._last_reselection_bar) >= effective_interval

    def should_reselect_adaptive(self, bar_count: int, current_regime: int = None,
                                 recent_regimes: list | None = None) -> bool:
        """Adaptive re-selection: increase cadence when regimes change rapidly.

        Parameters
        ----------
        bar_count: current bar index
        current_regime: latest regime label (optional)
        recent_regimes: list-like of recent regime labels (most-recent last)
        """
        # If no recent regime info, fall back to static cadence
        if recent_regimes is None or len(recent_regimes) < 2:
            return self.should_reselect(bar_count)

        # Measure regime switches in a recent window (last 20 labels or available)
        window = min(len(recent_regimes), 20)
        window_slice = recent_regimes[-window:]
        switches = sum(1 for i in range(1, len(window_slice)) if window_slice[i] != window_slice[i-1])

        # If more than 2 switches in the short window, consider regime rapidly changing
        if switches >= 2:
            # increase cadence (halve the interval, minimum 1)
            effective_interval = max(1, self.reselection_interval // 2)
        else:
            effective_interval = self.reselection_interval

        return (bar_count - self._last_reselection_bar) >= effective_interval

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

        new_pair_ids = set(
            new_pairs_df[["ticker1", "ticker2"]]
            .apply(lambda r: make_pair_id(r.ticker1, r.ticker2), axis=1)
            .tolist()
        )

        normalized_current = set()
        for pid in current_pair_ids:
            parts = split_pair_id(pid)
            if len(parts) == 2:
                normalized_current.add(make_pair_id(parts[0], parts[1]))

        added = new_pair_ids - normalized_current
        removed = normalized_current - new_pair_ids

        logger.info(
            "Pair re-selection #%d at bar %d: kept=%d, added=%d, removed=%d",
            self._reselection_count, bar_count,
            len(new_pair_ids & normalized_current), len(added), len(removed),
        )

        return new_pairs_df, added, removed

    @property
    def reselection_count(self) -> int:
        return self._reselection_count
