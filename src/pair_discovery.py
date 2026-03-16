"""Pair Discovery Engine.

Implements spec §4 — "for each regime: compute candidate pairs".

For each distinct regime label the engine:
  1. Slices the price matrix to dates belonging to that regime.
  2. Runs ``PairsSelector.find_pairs()`` (correlation pre-filter +
     Engle-Granger cointegration + half-life bounds).
  3. Tags results with the regime label.
  4. Collects all regime results into a single DataFrame with the spec
     output schema:
         pair_id | asset_A | asset_B | regime | coint_pvalue |
         hedge_ratio | half_life_days | spread_mean | spread_std |
         corr
"""

from __future__ import annotations

import logging
from typing import Optional, List
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from strategy.pairs_trading import PairsSelector
from utils.pair_id import make_pair_id

logger = logging.getLogger(__name__)


class PairDiscoveryEngine:
    """Discover cointegrated pairs within each detected regime.

    Parameters
    ----------
    pvalue_threshold : float
        Cointegration p-value cutoff (default 0.05).
    min_half_life : int
        Minimum mean-reversion half-life in days (default 5).
    max_half_life : int
        Maximum mean-reversion half-life in days (default 126 ≈ 6 months).
    max_pairs_per_regime : int
        Cap on pairs returned per regime (ranked by p-value).
    min_regime_bars : int
        Regimes with fewer bars than this are skipped (default 60).
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        min_half_life: int = 5,
        max_half_life: int = 126,
        max_pairs_per_regime: int = 50,
        min_regime_bars: int = 126,
    ):
        self.pvalue_threshold = pvalue_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.max_pairs_per_regime = max_pairs_per_regime
        self.min_regime_bars = max(126, int(min_regime_bars))


        self._selector = PairsSelector(
            pvalue_threshold=pvalue_threshold,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self,
        price_matrix: pd.DataFrame,
        regime_labels: pd.Series,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Run pair discovery for every regime and return combined results.

        Parameters
        ----------
        price_matrix : DataFrame
            Wide Date × Ticker close prices (from ``YFinanceClient.get_price_matrix``).
        regime_labels : Series
            Integer regime labels indexed by date
            (from ``HMMRegimeDetector.predict`` or ``fit_predict_walkforward``).
        tickers : list[str], optional
            Subset of tickers to consider.  Default = all columns of ``price_matrix``.

        Returns
        -------
        DataFrame with columns:
            pair_id, asset_A, asset_B, regime,
            coint_pvalue, hedge_ratio, half_life_days,
            spread_mean, spread_std, corr
        """
        # If user-supplied tickers include symbols not present in the price matrix,
        # filter them out rather than raising a KeyError (pandas: "[...] not in index").
        all_cols = list(price_matrix.columns)
        if tickers:
            missing = [t for t in tickers if t not in all_cols]
            if missing:
                logger.warning("Requested tickers not found in price matrix and will be ignored: %s", missing)
        tickers = [t for t in (tickers or all_cols) if t in all_cols]
        if not tickers:
            logger.warning("No valid tickers available for pair discovery after filtering."
                           " Ensure your price matrix contains the requested symbols.")
            return pd.DataFrame()

        # Align price matrix and regime labels on common dates
        common_dates = price_matrix.index.intersection(regime_labels.index)
        if common_dates.empty:
            logger.warning("No overlapping dates between price_matrix and regime_labels.")
            return pd.DataFrame()

        prices = price_matrix.loc[common_dates, tickers].copy()
        regimes = regime_labels.loc[common_dates]

        distinct_regimes = sorted(regimes.dropna().unique())
        logger.info("Running pair discovery across %d regimes: %s", len(distinct_regimes), distinct_regimes)

        def _discover_regime(regime_id: int) -> pd.DataFrame | None:
            regime_dates = regimes[regimes == regime_id].index
            regime_prices = prices.loc[regime_dates].dropna(axis=1, how="all")

            n_bars = len(regime_prices)
            if n_bars < self.min_regime_bars:
                logger.info(
                    "Regime %s: only %d bars (< %d required) — skipping.",
                    regime_id, n_bars, self.min_regime_bars,
                )
                return None
            if n_bars < 60:
                logger.warning(
                    "Regime %s: %d bars is below the minimum required for cointegration testing (60). "
                    "No pairs will be discovered.",
                    regime_id,
                    n_bars,
                )
                return None

            logger.info("Regime %s: %d bars, testing pairs...", regime_id, n_bars)

            pairs_df = self._selector.find_pairs(
                price_df=regime_prices,
                tickers=list(regime_prices.columns),
                max_pairs=self.max_pairs_per_regime,
                verbose=False,
                sequential_threshold=0,
            )

            if pairs_df.empty:
                logger.info("Regime %s: no cointegrated pairs found.", regime_id)
                return None

            # Add rolling pairwise correlation for each found pair
            pairs_df["corr"] = pairs_df.apply(
                lambda row: self._compute_pair_corr(regime_prices, row["ticker1"], row["ticker2"]),
                axis=1,
            )

            pairs_df["regime"] = regime_id
            return pairs_df

        n_cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 4)
        )
        regime_workers = max(1, min(len(distinct_regimes), n_cpus))
        with ThreadPoolExecutor(max_workers=regime_workers, thread_name_prefix="regime-discovery") as ex:
            discovered = list(ex.map(_discover_regime, distinct_regimes))

        all_results = [df for df in discovered if df is not None and not df.empty]

        if not all_results:
            logger.warning("No pairs found across any regime.")
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)
        combined = self._assign_pair_ids(combined)
        combined = self._standardize_columns(combined)

        logger.info(
            "Pair discovery complete: %d total (pair, regime) records across %d unique pairs.",
            len(combined),
            combined["pair_id"].nunique(),
        )
        return combined

    def discover_for_regime(
        self,
        price_matrix: pd.DataFrame,
        regime_id: int,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Convenience method: run discovery on a pre-sliced price matrix for one regime."""
        all_cols = list(price_matrix.columns)
        if tickers:
            missing = [t for t in tickers if t not in all_cols]
            if missing:
                logger.warning("Requested tickers not found in price matrix and will be ignored: %s", missing)
        tickers = [t for t in (tickers or all_cols) if t in all_cols]
        if not tickers:
            logger.warning("No valid tickers available for per-regime discovery after filtering.")
            return pd.DataFrame()
        pairs_df = self._selector.find_pairs(
            price_df=price_matrix[tickers],
            tickers=tickers,
            max_pairs=self.max_pairs_per_regime,
            verbose=False,
            sequential_threshold=0,
        )
        if pairs_df.empty:
            return pd.DataFrame()

        pairs_df["corr"] = pairs_df.apply(
            lambda row: self._compute_pair_corr(price_matrix, row["ticker1"], row["ticker2"]),
            axis=1,
        )
        pairs_df["regime"] = regime_id
        pairs_df = self._assign_pair_ids(pairs_df)
        return self._standardize_columns(pairs_df)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pair_corr(prices: pd.DataFrame, t1: str, t2: str) -> float:
        """Pearson correlation of daily returns between two tickers."""
        if t1 not in prices.columns or t2 not in prices.columns:
            return float("nan")
        rets = prices[[t1, t2]].pct_change().dropna()
        if len(rets) < 5:
            return float("nan")
        return round(float(rets[t1].corr(rets[t2])), 4)

    @staticmethod
    def _assign_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Create a canonical pair_id string 'AAAA–BBBB' (alphabetical order)."""
        df = df.copy()
        df["pair_id"] = df.apply(
            lambda r: make_pair_id(r["ticker1"], r["ticker2"]),
            axis=1,
        )
        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to the spec output schema and reorder."""
        df = df.rename(columns={
            "ticker1": "asset_A",
            "ticker2": "asset_B",
            "pvalue": "coint_pvalue",
        })
        ordered = [
            "pair_id", "asset_A", "asset_B", "regime",
            "coint_pvalue", "hedge_ratio", "half_life_days",
            "spread_mean", "spread_std", "corr",
        ]
        existing = [c for c in ordered if c in df.columns]
        extra = [c for c in df.columns if c not in ordered]
        return df[existing + extra].reset_index(drop=True)
