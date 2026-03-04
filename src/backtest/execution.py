"""Simulated broker / execution handler.

Models realistic execution costs:
    1. Bid-ask spread  — half-spread applied against fill direction
    2. Market-impact slippage — random adverse price move (proportional to order size)
    3. Commission — flat basis-points on notional

Short sales:
    All shorts are assumed fully marginable.  The broker records a short rebate
    at `short_rebate_rate` (annualised) credited daily by the Portfolio.

Parameters
----------
slippage_bps : float
    Fixed one-way slippage in basis points (applied in the adverse direction).
    E.g. 5 bps on a $100 stock = $0.05 adverse move.
spread_bps : float
    Half bid-ask spread in bps (cost of crossing the spread; 0.5 * full spread).
commission_pct : float
    Commission as a fraction of notional (e.g. 0.001 = 10 bps = $10 per $10k).
min_commission : float
    Minimum commission per fill in $.
short_rebate_rate : float
    Annualised % rate credited on short proceeds (e.g. 0.04 = 4% per year).
    Accrued and returned daily by Portfolio.call_daily_accrual().
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
import pandas as pd

from .events import FillEvent, OrderEvent, Direction


@dataclass
class ExecutionConfig:
    slippage_bps: float    = 5.0
    spread_bps: float      = 3.0
    commission_pct: float  = 0.001    # 10 bps
    min_commission: float  = 1.0      # $1 per fill
    short_rebate_rate: float = 0.04   # 4% p.a.
    random_slippage: bool = True      # add random noise on top of fixed slippage


class SimulatedBroker:
    """Fill orders at simulated realistic prices."""

    def __init__(self, config: Optional[ExecutionConfig] = None, seed: int = 42):
        self.cfg = config or ExecutionConfig()
        self._rng = np.random.default_rng(seed)
        self._fills: list[FillEvent] = []

    # -----------------------------------------------------------------------

    def execute(self, order: OrderEvent, prices: dict[str, float]) -> Optional[FillEvent]:
        """Execute an order against the current market snapshot.

        Returns a FillEvent, or None if the ticker has no price.
        """
        mid = prices.get(order.ticker)
        if mid is None or math.isnan(mid):
            return None

        qty    = order.quantity          # signed (+buy / -sell)
        is_buy = qty > 0

        # 1. Half-spread cost (always against the trader)
        spread_adj = mid * (self.cfg.spread_bps / 10_000)
        spread_cost = spread_adj * abs(qty)

        # 2. Fixed + optional random slippage
        slip_bps = self.cfg.slippage_bps
        if self.cfg.random_slippage:
            slip_bps += float(self._rng.exponential(scale=slip_bps * 0.5))
        slip_adj = mid * (slip_bps / 10_000)
        slippage_cost = slip_adj * abs(qty)

        # 3. Fill price: move adversely
        if is_buy:
            fill_price = mid + spread_adj + slip_adj
        else:
            fill_price = mid - spread_adj - slip_adj

        # 4. Commission
        notional   = abs(qty) * fill_price
        commission = max(self.cfg.min_commission, notional * self.cfg.commission_pct)

        fill = FillEvent(
            date         = order.date,
            ticker       = order.ticker,
            quantity     = qty,
            fill_price   = fill_price,
            commission   = commission,
            slippage_cost= spread_cost + slippage_cost,
            direction    = order.direction,
            strategy_id  = order.strategy_id,
            pair_id      = order.pair_id,
        )
        self._fills.append(fill)
        return fill

    # -----------------------------------------------------------------------

    def fills_df(self) -> pd.DataFrame:
        if not self._fills:
            return pd.DataFrame()
        rows = [
            {
                "date":          f.date,
                "ticker":        f.ticker,
                "quantity":      f.quantity,
                "fill_price":    f.fill_price,
                "notional":      f.notional,
                "commission":    f.commission,
                "slippage_cost": f.slippage_cost,
                "total_cost":    f.total_cost,
                "pair_id":       f.pair_id,
                "direction":     f.direction.value,
            }
            for f in self._fills
        ]
        return pd.DataFrame(rows).set_index("date")

    def total_commission(self) -> float:
        return sum(f.commission for f in self._fills)

    def total_slippage(self) -> float:
        return sum(f.slippage_cost for f in self._fills)
