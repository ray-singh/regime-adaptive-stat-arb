"""Portfolio accounting module.

Tracks:
    - Cash balance
    - Long/short equity positions (shares held)
    - Daily mark-to-market equity
    - Short-sale proceeds for rebate accrual
    - Trade log
    - Performance metrics (Sharpe, Sortino, drawdown, etc.)
"""

from __future__ import annotations

import math
from typing import Optional
import numpy as np
import pandas as pd

from .events import FillEvent, Direction


class Portfolio:
    """Mark-to-market portfolio with margin-aware position tracking.

    Parameters
    ----------
    initial_capital : float
        Starting cash ($).
    margin_requirement : float
        Fraction of short notional that must be held as cash collateral.
        E.g. 0.5 = 50% Reg-T margin.
    max_gross_leverage : float
        Hard cap on (longs + |shorts|) / equity.  Engine enforces via
        rejection of orders that would breach this.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        margin_requirement: float = 0.50,
        max_gross_leverage: float = 4.0,
    ):
        self.initial_capital   = initial_capital
        self.margin_requirement = margin_requirement
        self.max_gross_leverage = max_gross_leverage

        self.cash: float             = initial_capital
        # positions[ticker] = signed share count (+long / -short)
        self.positions: dict[str, float]     = {}
        self.avg_cost: dict[str, float]      = {}  # avg entry price per share (long > 0)
        self.short_proceeds: dict[str, float] = {}  # cash received from short sales

        # Historical records
        self._equity_curve: list[tuple[pd.Timestamp, float]] = []
        self._trades: list[dict] = []

        # Current market prices (updated by mark_to_market)
        self._last_prices: dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Core operations
    # -----------------------------------------------------------------------

    def update_fill(self, fill: FillEvent) -> None:
        """Apply a FillEvent to positions and cash."""
        t    = fill.ticker
        qty  = fill.quantity   # +buy / -sell
        cash_delta = fill.total_cost  # includes commission (negative = outflow)

        old_pos = self.positions.get(t, 0.0)
        new_pos = old_pos + qty

        # Track average cost basis for both long and short books.
        # For short inventory, avg_cost stores average short sale price.
        old_cost = self.avg_cost.get(t, fill.fill_price)
        if new_pos > 0:
            if old_pos > 0 and qty > 0:
                # Added to existing long.
                total_cost = old_pos * old_cost + qty * fill.fill_price
                self.avg_cost[t] = total_cost / new_pos
            elif old_pos <= 0 and new_pos > 0:
                # Crossed from flat/short into long; remaining long opened at this fill.
                self.avg_cost[t] = fill.fill_price
            # else: reduced long, keep prior cost basis.
        elif new_pos < 0:
            if old_pos < 0 and qty < 0:
                # Added to existing short (weighted by short shares).
                old_short = abs(old_pos)
                add_short = abs(qty)
                total_proceeds = old_short * old_cost + add_short * fill.fill_price
                self.avg_cost[t] = total_proceeds / abs(new_pos)
            elif old_pos >= 0 and new_pos < 0:
                # Crossed from flat/long into short; remaining short opened at this fill.
                self.avg_cost[t] = fill.fill_price
            # else: reduced short, keep prior short basis.

        # Short-sale proceeds tracking.
        # Only newly opened/increased short shares generate additional proceeds.
        if qty < 0:
            short_opened = 0.0
            if old_pos < 0:
                short_opened = abs(qty)  # extending existing short
            elif new_pos < 0:
                short_opened = abs(new_pos)  # crossed long -> short; only residual opens short

            if short_opened > 0:
                proceeds = short_opened * fill.fill_price
                self.short_proceeds[t] = self.short_proceeds.get(t, 0.0) + proceeds

        if qty > 0 and old_pos < 0:
            # Covering short — release short proceeds proportionally.
            covered = min(qty, abs(old_pos))
            if covered > 0:
                current_proceeds = self.short_proceeds.get(t, 0.0)
                cover_frac = covered / abs(old_pos)
                self.short_proceeds[t] = max(0.0, current_proceeds * (1 - cover_frac))

        self.positions[t] = new_pos
        if abs(new_pos) < 1e-12:
            self.positions.pop(t, None)
            self.avg_cost.pop(t, None)
            self.short_proceeds.pop(t, None)

        self.cash += cash_delta

        # Record trade
        self._trades.append({
            "date":        fill.date,
            "ticker":      fill.ticker,
            "quantity":    fill.quantity,
            "fill_price":  fill.fill_price,
            "commission":  fill.commission,
            "slippage":    fill.slippage_cost,
            "cash_delta":  cash_delta,
            "pair_id":     fill.pair_id,
        })

    def mark_to_market(self, date: pd.Timestamp, prices: dict[str, float]) -> float:
        """Update equity curve with current prices.  Returns total equity."""
        self._last_prices.update(prices)
        equity = self.total_equity(prices)
        self._equity_curve.append((date, equity))
        return equity

    def accrue_short_rebate(self, rebate_rate_annual: float = 0.04) -> None:
        """Credit daily short rebate on short proceeds (call once per day)."""
        daily_rate = rebate_rate_annual / 252
        for t, proceeds in self.short_proceeds.items():
            self.cash += proceeds * daily_rate

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def total_equity(self, prices: Optional[dict[str, float]] = None) -> float:
        """Market value of portfolio (cash + long MV − short MV)."""
        p = prices or self._last_prices
        pos_mv = sum(
            shares * p.get(t, self.avg_cost.get(t, 0.0))
            for t, shares in self.positions.items()
        )
        return self.cash + pos_mv

    def gross_leverage(self, prices: Optional[dict[str, float]] = None) -> float:
        p = prices or self._last_prices
        equity = self.total_equity(p)
        if equity <= 0:
            return float("inf")
        gross_notional = sum(
            abs(shares) * p.get(t, self.avg_cost.get(t, 0.0))
            for t, shares in self.positions.items()
        )
        return gross_notional / equity

    def net_leverage(self, prices: Optional[dict[str, float]] = None) -> float:
        p = prices or self._last_prices
        equity = self.total_equity(p)
        if equity <= 0:
            return float("inf")
        net_notional = sum(
            shares * p.get(t, self.avg_cost.get(t, 0.0))
            for t, shares in self.positions.items()
        )
        return net_notional / equity

    def margin_available(self) -> float:
        """Cash not pledged as short margin collateral."""
        pledged = sum(
            self.margin_requirement * proceeds
            for proceeds in self.short_proceeds.values()
        )
        return self.cash - pledged

    def get_position(self, ticker: str) -> float:
        return self.positions.get(ticker, 0.0)

    # -----------------------------------------------------------------------
    # Results / Reporting
    # -----------------------------------------------------------------------

    def equity_curve(self) -> pd.Series:
        if not self._equity_curve:
            return pd.Series(dtype=float)
        dates, vals = zip(*self._equity_curve)
        return pd.Series(vals, index=dates, name="equity")

    def trades_df(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame(self._trades).set_index("date")

    def performance_stats(self, risk_free_rate: float = 0.05) -> dict:
        eq = self.equity_curve()
        if len(eq) < 2:
            return {}

        rets  = eq.pct_change().dropna()
        daily_rf = risk_free_rate / 252
        excess   = rets - daily_rf

        # Annualised return (CAGR)
        n_years  = len(rets) / 252
        end_val  = eq.iloc[-1]
        cagr     = (end_val / self.initial_capital) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Sharpe
        sharpe   = excess.mean() / excess.std() * math.sqrt(252) if excess.std() > 0 else float("nan")

        # Sortino (downside deviation)
        neg      = excess[excess < 0]
        sortino  = excess.mean() / neg.std() * math.sqrt(252) if len(neg) > 1 and neg.std() > 0 else float("nan")

        # Calmar
        roll_max = eq.cummax()
        dd       = (eq - roll_max) / roll_max
        max_dd   = dd.min()
        calmar   = cagr / abs(max_dd) if max_dd != 0 else float("nan")

        # Win rate
        trades = self.trades_df()
        n_trades = len(trades)

        total_commission = trades["commission"].sum() if "commission" in trades.columns else 0.0
        total_slippage   = trades["slippage"].sum()   if "slippage"   in trades.columns else 0.0

        return {
            "initial_capital":    self.initial_capital,
            "final_equity":       round(end_val, 2),
            "total_return_pct":   round((end_val / self.initial_capital - 1) * 100, 2),
            "cagr_pct":           round(cagr * 100, 2),
            "ann_vol_pct":        round(rets.std() * math.sqrt(252) * 100, 2),
            "sharpe_ratio":       round(sharpe, 3),
            "sortino_ratio":      round(sortino, 3),
            "calmar_ratio":       round(calmar, 3),
            "max_drawdown_pct":   round(max_dd * 100, 2),
            "n_trades":           n_trades,
            "total_commission":   round(total_commission, 2),
            "total_slippage":     round(total_slippage, 2),
            "trading_days":       len(rets),
        }
