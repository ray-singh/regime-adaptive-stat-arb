"""Event types for the event-driven backtest engine.

Event flow:
    MarketEvent → (Strategy) → SignalEvent
    SignalEvent → (PositionSizer) → OrderEvent
    OrderEvent  → (Broker/ExecutionHandler) → FillEvent
    FillEvent   → (Portfolio) → updated positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import pandas as pd


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER  = auto()
    FILL   = auto()


class Direction(str, Enum):
    LONG       = "LONG"       # buy
    SHORT      = "SHORT"      # sell
    FLAT       = "FLAT"       # close / exit


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


# ---------------------------------------------------------------------------

@dataclass
class MarketEvent:
    """Fired once per bar for each ticker that has a new price."""
    event_type: EventType = field(default=EventType.MARKET, init=False)
    date: pd.Timestamp
    prices: dict[str, float]          # {ticker: close_price}
    ohlcv: Optional[dict] = None      # full bar data if available


@dataclass
class SignalEvent:
    """Strategy-generated directional signal for a pair spread."""
    event_type: EventType = field(default=EventType.SIGNAL, init=False)
    date: pd.Timestamp
    ticker1: str
    ticker2: str
    direction: str                    # "long_spread" | "short_spread" | "flat"
    strength: float = 1.0             # 0–1 scalar, used for position sizing
    spread_zscore: float = 0.0
    hedge_ratio: float = 1.0
    strategy_id: str = "pairs"


@dataclass
class OrderEvent:
    """Single-leg order to execute."""
    event_type: EventType = field(default=EventType.ORDER, init=False)
    date: pd.Timestamp
    ticker: str
    order_type: OrderType
    quantity: float                   # +ve = buy, -ve = sell (shares or $ notional)
    direction: Direction
    limit_price: Optional[float] = None
    strategy_id: str = "pairs"
    pair_id: str = ""                 # e.g. "AAPL/AMD"


@dataclass
class FillEvent:
    """Execution confirmation for a single-leg order."""
    event_type: EventType = field(default=EventType.FILL, init=False)
    date: pd.Timestamp
    ticker: str
    quantity: float                   # signed: +ve = bought, -ve = sold
    fill_price: float                 # actual execution price (incl. slippage)
    commission: float                 # total commission $ for this fill
    slippage_cost: float              # $ slippage vs mid
    direction: Direction
    strategy_id: str = "pairs"
    pair_id: str = ""

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.fill_price

    @property
    def total_cost(self) -> float:
        """Net cash impact (negative = cash outflow)."""
        return -(self.quantity * self.fill_price) - self.commission
