"""Event-driven backtesting engine for regime-adaptive statistical arbitrage."""

from .events import MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
from .data_feed import HistoricalDataFeed
from .execution import SimulatedBroker
from .portfolio import Portfolio
from .engine import BacktestEngine
from .strategy_wrapper import PairsBacktestStrategy

__all__ = [
    "MarketEvent", "SignalEvent", "OrderEvent", "FillEvent", "EventType",
    "HistoricalDataFeed", "SimulatedBroker", "Portfolio", "BacktestEngine",
    "PairsBacktestStrategy",
]
