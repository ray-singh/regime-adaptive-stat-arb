"""Pairs Trading Strategy.

Workflow:
    1. PairsSelector.find_pairs() — Engle-Granger cointegration test on all pairs
    2. PairsTradingStrategy.generate_signals() — rolling z-score of spread
    3. Evaluate() — basic P&L, Sharpe, hit-rate per regime

Signal rules:
    Entry long spread  (buy leg1, sell leg2): z < -entry_z
    Entry short spread (sell leg1, buy leg2): z >  entry_z
    Exit:                                      |z| <  exit_z
    Stop:                                      |z| >  stop_z
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Optional, Tuple, Dict
import logging
import warnings

from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pair Selection
# ─────────────────────────────────────────────────────────────────────────────

class PairsSelector:
    """Find cointegrated pairs using Engle-Granger test.

    Parameters
    ----------
    pvalue_threshold : float
        Maximum p-value for cointegration (default 0.05).
    min_half_life : int
        Minimum mean-reversion half-life in days.
    max_half_life : int
        Maximum mean-reversion half-life in days.
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        min_half_life: int = 5,
        max_half_life: int = 126,
    ):
        self.pvalue_threshold = pvalue_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def find_pairs(
        self,
        price_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        max_pairs: int = 50,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Find cointegrated pairs in a wide price DataFrame.

        Parameters
        ----------
        price_df : DataFrame
            Wide price matrix (index=Date, columns=tickers, values=adj_close/close).
        tickers : list[str], optional
            Subset of tickers to test. Default = all columns.
        max_pairs : int
            Maximum pairs to return (ranked by p-value).

        Returns
        -------
        DataFrame with columns: ticker1, ticker2, pvalue, hedge_ratio, half_life_days
        """
        tickers = tickers or list(price_df.columns)
        pairs_result = []

        combos = list(combinations(tickers, 2))
        logger.info(f"Testing {len(combos)} pairs for cointegration...")

        for i, (t1, t2) in enumerate(combos):
            if i % 200 == 0 and verbose:
                logger.info(f"  {i}/{len(combos)} tested, {len(pairs_result)} found so far")

            s1 = price_df[t1].dropna()
            s2 = price_df[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 252:
                continue

            s1, s2 = s1[common], s2[common]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    score, pvalue, _ = coint(s1, s2)
                except Exception:
                    continue

            if pvalue > self.pvalue_threshold:
                continue

            hedge_ratio = self._estimate_hedge_ratio(s1, s2)
            spread = s1 - hedge_ratio * s2
            half_life = self._half_life(spread)

            if not (self.min_half_life <= half_life <= self.max_half_life):
                continue

            pairs_result.append({
                "ticker1": t1,
                "ticker2": t2,
                "pvalue": round(pvalue, 5),
                "hedge_ratio": round(hedge_ratio, 4),
                "half_life_days": round(half_life, 1),
                "spread_mean": round(spread.mean(), 4),
                "spread_std": round(spread.std(), 4),
            })

        result = pd.DataFrame(pairs_result)
        if not result.empty:
            result = (
                result.sort_values("pvalue")
                .head(max_pairs)
                .reset_index(drop=True)
            )
        logger.info(f"Found {len(result)} cointegrated pairs")
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_hedge_ratio(s1: pd.Series, s2: pd.Series) -> float:
        """OLS hedge ratio: s1 = β * s2 + α."""
        X = add_constant(s2.values)
        res = OLS(s1.values, X).fit()
        return float(res.params[1])

    @staticmethod
    def _half_life(spread: pd.Series) -> float:
        """Ornstein-Uhlenbeck half-life via AR(1) regression."""
        lagged = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        common = lagged.index.intersection(delta.index)
        lagged, delta = lagged[common], delta[common]
        X = add_constant(lagged.values)
        res = OLS(delta.values, X).fit()
        theta = -res.params[1]
        if theta <= 0:
            return np.inf
        return np.log(2) / theta


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ─────────────────────────────────────────────────────────────────────────────

class PairsTradingStrategy:
    """Generate long/short spread signals via rolling z-score.

    Parameters
    ----------
    zscore_window : int
        Rolling window for z-score computation.
    entry_z : float
        |z| threshold to open a position.
    exit_z : float
        |z| below which to close.
    stop_z : float
        |z| above which to stop-loss.
    """

    def __init__(
        self,
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ):
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def compute_spread(
        self,
        price_df: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        hedge_ratio: float,
    ) -> pd.Series:
        """Spread = price1 - hedge_ratio * price2."""
        s1 = price_df[ticker1]
        s2 = price_df[ticker2]
        return (s1 - hedge_ratio * s2).rename(f"{ticker1}/{ticker2}")

    def compute_zscore(self, spread: pd.Series) -> pd.Series:
        mu = spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2).mean()
        sigma = spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2).std()
        return ((spread - mu) / sigma.replace(0, np.nan)).rename("zscore")

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        hedge_ratio: float,
        regime_labels: Optional[pd.Series] = None,
        active_regimes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate daily signals for a single pair.

        Returns a DataFrame with columns:
            spread, zscore, position, pnl_daily, pnl_cumulative
        where position is:
            +1 = long spread (buy t1, sell t2)
            -1 = short spread (sell t1, buy t2)
             0 = flat
        """
        spread = self.compute_spread(price_df, ticker1, ticker2, hedge_ratio)
        zscore = self.compute_zscore(spread)

        # Raw returns of each leg
        ret1 = price_df[ticker1].pct_change()
        ret2 = price_df[ticker2].pct_change()

        # Build position series via state machine
        position = pd.Series(0, index=spread.index, dtype=int)
        pos = 0
        for i in range(1, len(zscore)):
            z = zscore.iloc[i]
            if pd.isna(z):
                position.iloc[i] = 0
                pos = 0
                continue

            # Check regime at this timestep — if regime filter is active, block new entries
            # (but always allow exits to avoid being locked in a losing position)
            in_active_regime = True
            if regime_labels is not None and active_regimes is not None:
                try:
                    r = regime_labels[zscore.index[i]]
                    in_active_regime = r in active_regimes
                except (KeyError, TypeError):
                    in_active_regime = True  # unknown regime → allow trading

            # Exit / stop conditions (always evaluated regardless of regime)
            if pos != 0:
                if abs(z) < self.exit_z or abs(z) > self.stop_z:
                    pos = 0

            # Entry conditions (gated by regime)
            if pos == 0 and in_active_regime:
                if z < -self.entry_z:
                    pos = 1    # long spread
                elif z > self.entry_z:
                    pos = -1   # short spread

            position.iloc[i] = pos

        # P&L: long spread = long t1 + short t2, scaled by hedge ratio
        pnl = position.shift(1) * (ret1 - hedge_ratio * ret2)

        result = pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "pnl_daily": pnl,
            "pnl_cumulative": pnl.cumsum(),
        })
        return result

    def evaluate(self, signals_df: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict:
        """Compute summary statistics for a signals DataFrame."""
        pnl = signals_df["pnl_daily"].dropna()
        cum = signals_df["pnl_cumulative"].dropna()

        n_trades = (signals_df["position"].diff().abs() > 0).sum()
        active_days = (signals_df["position"] != 0).sum()

        daily_rf = risk_free_rate / 252
        excess = pnl - daily_rf
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan

        # Max drawdown
        roll_max = cum.cummax()
        dd = cum - roll_max
        max_dd = dd.min()

        # Win rate on active days
        wins = (pnl[signals_df["position"].shift(1) != 0] > 0).mean()

        return {
            "total_return_pct": round(cum.iloc[-1] * 100, 2) if len(cum) else np.nan,
            "ann_return_pct": round(pnl.mean() * 252 * 100, 2),
            "ann_vol_pct": round(pnl.std() * np.sqrt(252) * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_trades": int(n_trades),
            "active_days": int(active_days),
            "win_rate": round(wins, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a wide price matrix from long-format OHLCV DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_price_matrix(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Pivot long-format OHLCV df → wide (Date × ticker) price matrix."""
    if price_col not in df.columns:
        price_col = "close"
    # Ensure Date is tz-naive
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    wide = df.pivot(index="Date", columns="ticker", values=price_col)
    return wide.sort_index()
