"""Relationship Analysis Layer — spec §5.

For each (pair, regime) record from the Pair Discovery Engine, compute:
  - correlation_stability  : rolling-corr std across the regime window
  - spread_variance        : variance of the spread series in that regime
  - half_life              : OU half-life (already from discovery; refined here per-regime)
  - regime_dependency      : flag indicating the pair behaves differently across regimes

Produces two output tables:
  1. ``pair_regime_stats``  — one row per (pair_id, regime)
  2. ``pair_summary``       — one row per pair_id:
       - stable_regimes   : list of regime ids where coint_pvalue < threshold
       - unstable_regimes : list of regime ids where coint_pvalue >= threshold
       - n_regimes_active : count of regimes where pair was found
       - regime_sensitive : True if the pair varies significantly across regimes
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A pair is "stable" in a regime when its cointegration p-value is below this
_STABLE_PVALUE_THRESHOLD = 0.05

# A pair is "regime-sensitive" when its corr varies more than this across regimes
_REGIME_SENSITIVITY_CORR_STD = 0.15


class RelationshipAnalyzer:
    """Analyse how pair relationships change across regimes.

    Parameters
    ----------
    price_matrix : DataFrame
        Wide Date × Ticker close prices — used to compute per-regime spread stats.
    stable_pvalue : float
        Cointegration p-value below which a pair is considered stable in a regime.
    sensitivity_corr_std : float
        If the std of per-regime correlations exceeds this, the pair is flagged
        as regime-sensitive.
    corr_window : int
        Rolling window (bars) for within-regime rolling correlation.
    """

    def __init__(
        self,
        price_matrix: pd.DataFrame,
        stable_pvalue: float = _STABLE_PVALUE_THRESHOLD,
        sensitivity_corr_std: float = _REGIME_SENSITIVITY_CORR_STD,
        corr_window: int = 20,
    ):
        self.price_matrix = price_matrix
        self.stable_pvalue = stable_pvalue
        self.sensitivity_corr_std = sensitivity_corr_std
        self.corr_window = corr_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        pairs_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute relationship stats for all pairs across all regimes.

        Parameters
        ----------
        pairs_df : DataFrame
            Output of ``PairDiscoveryEngine.discover()`` — must contain
            pair_id, asset_A, asset_B, regime, coint_pvalue, hedge_ratio,
            half_life_days, corr.
        regime_labels : Series
            Integer regime labels indexed by date.

        Returns
        -------
        pair_regime_stats : DataFrame
            Per (pair_id, regime) statistics.
        pair_summary : DataFrame
            Per pair_id aggregated summary (stable/unstable regimes, sensitivity).
        """
        if pairs_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        regime_stats_rows: list[dict] = []

        for _, row in pairs_df.iterrows():
            stats = self._compute_regime_stats(row, regime_labels)
            regime_stats_rows.append(stats)

        pair_regime_stats = pd.DataFrame(regime_stats_rows)
        pair_summary = self._build_pair_summary(pair_regime_stats)

        return pair_regime_stats, pair_summary

    # ------------------------------------------------------------------
    # Per-(pair, regime) statistics
    # ------------------------------------------------------------------

    def _compute_regime_stats(self, row: pd.Series, regime_labels: pd.Series) -> dict:
        """Compute stats for one (pair, regime) slice."""
        a, b = row["asset_A"], row["asset_B"]
        regime_id = row["regime"]
        hedge_ratio = row.get("hedge_ratio", 1.0)

        # Dates in this regime
        regime_dates = regime_labels[regime_labels == regime_id].index
        common = self.price_matrix.index.intersection(regime_dates)

        base = {
            "pair_id": row["pair_id"],
            "asset_A": a,
            "asset_B": b,
            "regime": regime_id,
            "coint_pvalue": row.get("coint_pvalue", np.nan),
            "hedge_ratio": hedge_ratio,
            "half_life_days": row.get("half_life_days", np.nan),
            "corr": row.get("corr", np.nan),
        }

        if common.empty or a not in self.price_matrix.columns or b not in self.price_matrix.columns:
            base.update({
                "spread_variance": np.nan,
                "corr_stability": np.nan,
                "is_stable": False,
            })
            return base

        slice_prices = self.price_matrix.loc[common, [a, b]].dropna()
        if len(slice_prices) < 10:
            base.update({"spread_variance": np.nan, "corr_stability": np.nan, "is_stable": False})
            return base

        # Spread series: s_A - hedge_ratio * s_B
        spread = slice_prices[a] - hedge_ratio * slice_prices[b]
        spread_variance = float(spread.var())

        # Correlation stability: std of rolling correlation (lower = more stable)
        rets = slice_prices.pct_change().dropna()
        if len(rets) >= self.corr_window:
            rolling_corr = rets[a].rolling(self.corr_window, min_periods=self.corr_window // 2).corr(rets[b])
            corr_stability = float(rolling_corr.std())
        else:
            corr_stability = float(rets[a].corr(rets[b]))  # fallback: scalar

        is_stable = row.get("coint_pvalue", 1.0) < self.stable_pvalue

        base.update({
            "spread_variance": round(spread_variance, 6),
            "corr_stability": round(corr_stability, 4),
            "is_stable": is_stable,
        })
        return base

    # ------------------------------------------------------------------
    # Pair-level summary
    # ------------------------------------------------------------------

    def _build_pair_summary(self, pair_regime_stats: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-regime stats into a per-pair summary table."""
        if pair_regime_stats.empty:
            return pd.DataFrame()

        rows = []
        for pair_id, grp in pair_regime_stats.groupby("pair_id"):
            stable_mask = grp["is_stable"]
            stable_regimes = sorted(grp.loc[stable_mask, "regime"].tolist())
            unstable_regimes = sorted(grp.loc[~stable_mask, "regime"].tolist())

            # Regime sensitivity: std of per-regime correlation
            corr_vals = grp["corr"].dropna()
            regime_corr_std = float(corr_vals.std()) if len(corr_vals) > 1 else 0.0
            regime_sensitive = regime_corr_std > self.sensitivity_corr_std

            # Average spread variance across all regimes  
            avg_spread_var = float(grp["spread_variance"].mean())

            # Best (lowest) half-life and cointegration p-value across all regimes
            best_half_life = float(grp["half_life_days"].min())
            best_pvalue = float(grp["coint_pvalue"].min())

            rows.append({
                "pair_id": pair_id,
                "asset_A": grp["asset_A"].iloc[0],
                "asset_B": grp["asset_B"].iloc[0],
                "n_regimes_active": len(grp),
                "stable_regimes": stable_regimes,
                "unstable_regimes": unstable_regimes,
                "regime_sensitive": regime_sensitive,
                "regime_corr_std": round(regime_corr_std, 4),
                "avg_spread_variance": round(avg_spread_var, 6),
                "best_half_life_days": round(best_half_life, 1),
                "best_coint_pvalue": round(best_pvalue, 5),
            })

        summary = pd.DataFrame(rows).sort_values("best_coint_pvalue").reset_index(drop=True)
        return summary
