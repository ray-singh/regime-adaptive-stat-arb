"""Pair Ranking Engine — spec §6.

Scores pairs by "interestingness" using three criteria:

  stability_score      = 1 - (unstable_regimes / total_regimes_tested)
                         High → pair is consistently cointegrated.

  regime_sensitivity   = regime_corr_std (std of per-regime correlation)
                         High → pair relationship changes dramatically across regimes
                         (interesting for regime-conditional trading).

  mean_reversion_str   = 1 / best_half_life_days  (normalised to [0, 1])
                         High → spread reverts quickly.

Final score (spec formula):
    score = w_regime_var * regime_sensitivity + w_mr * mean_reversion_str

Stability score is also reported so the dashboard can display both.

Output schema (one row per pair_id, sorted by score descending):
    pair_id | asset_A | asset_B |
    stability_score | regime_sensitivity | mean_reversion_str |
    score | rank |
    stable_regimes | unstable_regimes | regime_sensitive |
    best_half_life_days | best_coint_pvalue
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PairRankingEngine:
    """Score and rank pairs by interestingness.

    Parameters
    ----------
    w_regime_sensitivity : float
        Weight for regime sensitivity component (default 0.6).
    w_mean_reversion : float
        Weight for mean-reversion strength component (default 0.4).
    max_half_life_days : float
        Half-life ceiling used for normalisation (default 126 days ≈ 6 months).
    """

    def __init__(
        self,
        w_regime_sensitivity: float = 0.6,
        w_mean_reversion: float = 0.4,
        max_half_life_days: float = 126.0,
    ):
        self.w_regime_sensitivity = w_regime_sensitivity
        self.w_mean_reversion = w_mean_reversion
        self.max_half_life_days = max_half_life_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rank(self, pair_summary: pd.DataFrame) -> pd.DataFrame:
        """Score and rank pairs from the RelationshipAnalyzer summary table.

        Parameters
        ----------
        pair_summary : DataFrame
            Output of ``RelationshipAnalyzer.analyse()``  (the second return value).

        Returns
        -------
        DataFrame sorted by ``score`` descending with an added ``rank`` column.
        """
        if pair_summary.empty:
            return pd.DataFrame()

        df = pair_summary.copy()

        # --- stability_score: fraction of regimes where pair was stable ---
        df["total_regimes"] = df["n_regimes_active"]
        df["n_stable"] = df["stable_regimes"].apply(len)
        df["stability_score"] = (df["n_stable"] / df["total_regimes"].replace(0, np.nan)).fillna(0.0)

        # --- regime_sensitivity: already computed as regime_corr_std, normalise to [0,1] ---
        max_cs = df["regime_corr_std"].max()
        if max_cs > 0:
            df["regime_sensitivity"] = (df["regime_corr_std"] / max_cs).clip(0, 1)
        else:
            df["regime_sensitivity"] = 0.0

        # --- mean_reversion_str: 1/half_life, normalised ---
        hl = df["best_half_life_days"].clip(lower=1.0)
        mr_raw = 1.0 / hl
        max_mr = mr_raw.max()
        df["mean_reversion_str"] = (mr_raw / max_mr).clip(0, 1) if max_mr > 0 else 0.0

        # --- final score (spec §6) ---
        df["score"] = (
            self.w_regime_sensitivity * df["regime_sensitivity"]
            + self.w_mean_reversion * df["mean_reversion_str"]
        ).round(4)

        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        # --- human-readable stability label ---
        df["stability_label"] = pd.cut(
            df["stability_score"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["low", "medium", "high"],
        )

        # Reorder columns for clean output
        front = [
            "rank", "pair_id", "asset_A", "asset_B",
            "score", "stability_score", "stability_label",
            "regime_sensitivity", "mean_reversion_str",
            "stable_regimes", "unstable_regimes", "regime_sensitive",
            "best_half_life_days", "best_coint_pvalue",
        ]
        present = [c for c in front if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        return df[present + rest].reset_index(drop=True)

    def top_n(self, pair_summary: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return the top-N ranked pairs."""
        ranked = self.rank(pair_summary)
        return ranked.head(n)
