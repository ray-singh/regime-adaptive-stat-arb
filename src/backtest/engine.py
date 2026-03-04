"""Event-driven backtesting engine.

Event flow (repeated once per trading day):
    1. DataFeed produces a MarketEvent
    2. Strategy receives the MarketEvent, appends to its price history,
       and emits zero or more SignalEvents (one per pair)
    3. PositionSizer converts each SignalEvent into OrderEvents (two legs)
    4. SimulatedBroker executes each OrderEvent → FillEvents
    5. Portfolio updates positions / cash for each FillEvent
    6. Portfolio marks equity to market
"""

from __future__ import annotations

import queue
import logging
from typing import Optional
import pandas as pd

from .events import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent
from .data_feed import HistoricalDataFeed
from .execution import SimulatedBroker
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Drives the event loop.

    Parameters
    ----------
    data_feed : HistoricalDataFeed
    strategy : any object implementing on_market_event(event) → list[SignalEvent]
    portfolio : Portfolio
    broker : SimulatedBroker
    position_sizer : callable(signal, portfolio, prices) → list[OrderEvent]
        Default: `default_position_sizer` below.
    short_rebate_rate : float
        Annual rate passed to Portfolio.accrue_short_rebate each day.
    verbose : bool
    """

    def __init__(
        self,
        data_feed: HistoricalDataFeed,
        strategy,
        portfolio: Portfolio,
        broker: SimulatedBroker,
        position_sizer=None,
        short_rebate_rate: float = 0.04,
        verbose: bool = False,
    ):
        self.data_feed         = data_feed
        self.strategy          = strategy
        self.portfolio         = portfolio
        self.broker            = broker
        self.position_sizer    = position_sizer or default_position_sizer
        self.short_rebate_rate = short_rebate_rate
        self.verbose           = verbose

        self._event_queue: queue.Queue = queue.Queue()
        self._bars_processed   = 0
        self._signals_fired    = 0
        self._orders_sent      = 0
        self._fills_received   = 0

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        """Run the backtest end-to-end and return a results dict."""
        logger.info("Backtest started — %d trading days in feed", len(self.data_feed))

        for market_event in self.data_feed:
            # ── 1. Push market event ────────────────────────────────────────
            self._event_queue.put(market_event)

            # ── 2. Drain queue for this bar ──────────────────────────────────
            while not self._event_queue.empty():
                event = self._event_queue.get(block=False)

                if event.event_type == EventType.MARKET:
                    self._handle_market(event)

                elif event.event_type == EventType.SIGNAL:
                    self._handle_signal(event)

                elif event.event_type == EventType.ORDER:
                    self._handle_order(event)

                elif event.event_type == EventType.FILL:
                    self._handle_fill(event)

            # ── 3. End-of-day portfolio maintenance ──────────────────────────
            prices = self.data_feed.current_prices()
            self.portfolio.mark_to_market(market_event.date, prices)
            self.portfolio.accrue_short_rebate(self.short_rebate_rate)

            self._bars_processed += 1
            if self.verbose and self._bars_processed % 252 == 0:
                eq = self.portfolio.total_equity(prices)
                logger.info(
                    "Bar %d | %s | Equity: $%.0f | Positions: %d",
                    self._bars_processed,
                    market_event.date.date(),
                    eq,
                    len(self.portfolio.positions),
                )

        stats = self.portfolio.performance_stats()
        logger.info("Backtest complete — %d bars, %d fills", self._bars_processed, self._fills_received)
        logger.info("Performance: %s", {k: v for k, v in stats.items() if k in
                    ("cagr_pct", "sharpe_ratio", "max_drawdown_pct", "n_trades")})

        return {
            "stats":        stats,
            "equity_curve": self.portfolio.equity_curve(),
            "trades":       self.portfolio.trades_df(),
            "fills":        self.broker.fills_df(),
            "bars":         self._bars_processed,
            "signals":      self._signals_fired,
            "orders":       self._orders_sent,
            "fills_count":  self._fills_received,
        }

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def _handle_market(self, event: MarketEvent) -> None:
        signals = self.strategy.on_market_event(event, self.portfolio)
        for sig in (signals or []):
            self._event_queue.put(sig)
            self._signals_fired += 1

    def _handle_signal(self, event: SignalEvent) -> None:
        prices = self.data_feed.current_prices()
        orders = self.position_sizer(event, self.portfolio, prices)
        for order in (orders or []):
            self._event_queue.put(order)
            self._orders_sent += 1

    def _handle_order(self, event: OrderEvent) -> None:
        prices = self.data_feed.current_prices()
        fill   = self.broker.execute(event, prices)
        if fill is not None:
            self._event_queue.put(fill)

    def _handle_fill(self, event: FillEvent) -> None:
        self.portfolio.update_fill(event)
        self._fills_received += 1


# ---------------------------------------------------------------------------
# Default position sizer
# ---------------------------------------------------------------------------

def default_position_sizer(
    signal: SignalEvent,
    portfolio: Portfolio,
    prices: dict[str, float],
    target_notional_pct: float = 0.10,   # 10% of equity per pair leg
) -> list[OrderEvent]:
    """Convert a SignalEvent into two-leg OrderEvents.

    Direction of legs:
        long_spread  → buy ticker1, sell ticker2 (hedge_ratio-weighted)
        short_spread → sell ticker1, buy ticker2
        flat         → close any open positions in both legs
    """
    from .events import Direction, OrderType, OrderEvent

    t1, t2  = signal.ticker1, signal.ticker2
    hr      = signal.hedge_ratio
    pair_id = f"{t1}/{t2}"

    equity  = portfolio.total_equity(prices)
    target  = equity * target_notional_pct * signal.strength

    p1 = prices.get(t1)
    p2 = prices.get(t2)
    if p1 is None or p2 is None:
        return []

    orders: list[OrderEvent] = []

    if signal.direction == "flat":
        # Close both legs
        pos1 = portfolio.get_position(t1)
        pos2 = portfolio.get_position(t2)
        if pos1 != 0:
            orders.append(OrderEvent(
                date=signal.date, ticker=t1, order_type=OrderType.MARKET,
                quantity=-pos1,   # close
                direction=Direction.FLAT, strategy_id=signal.strategy_id, pair_id=pair_id,
            ))
        if pos2 != 0:
            orders.append(OrderEvent(
                date=signal.date, ticker=t2, order_type=OrderType.MARKET,
                quantity=-pos2,
                direction=Direction.FLAT, strategy_id=signal.strategy_id, pair_id=pair_id,
            ))
        return orders

    # Long or short spread
    shares1 = round(target / p1, 4)
    shares2 = round(target * hr / p2, 4)

    if signal.direction == "long_spread":
        # Buy t1, sell t2
        # First close any existing position, then open
        cur1 = portfolio.get_position(t1)
        cur2 = portfolio.get_position(t2)

        # Only send orders if there would be a meaningful change
        if abs(shares1 - cur1) > 0.01:
            net1 = shares1 - cur1
            orders.append(OrderEvent(
                date=signal.date, ticker=t1, order_type=OrderType.MARKET,
                quantity=net1, direction=Direction.LONG if net1 > 0 else Direction.SHORT,
                strategy_id=signal.strategy_id, pair_id=pair_id,
            ))
        if abs(-shares2 - cur2) > 0.01:
            net2 = -shares2 - cur2
            orders.append(OrderEvent(
                date=signal.date, ticker=t2, order_type=OrderType.MARKET,
                quantity=net2, direction=Direction.SHORT if net2 < 0 else Direction.LONG,
                strategy_id=signal.strategy_id, pair_id=pair_id,
            ))

    elif signal.direction == "short_spread":
        # Sell t1, buy t2
        cur1 = portfolio.get_position(t1)
        cur2 = portfolio.get_position(t2)

        if abs(-shares1 - cur1) > 0.01:
            net1 = -shares1 - cur1
            orders.append(OrderEvent(
                date=signal.date, ticker=t1, order_type=OrderType.MARKET,
                quantity=net1, direction=Direction.SHORT if net1 < 0 else Direction.LONG,
                strategy_id=signal.strategy_id, pair_id=pair_id,
            ))
        if abs(shares2 - cur2) > 0.01:
            net2 = shares2 - cur2
            orders.append(OrderEvent(
                date=signal.date, ticker=t2, order_type=OrderType.MARKET,
                quantity=net2, direction=Direction.LONG if net2 > 0 else Direction.SHORT,
                strategy_id=signal.strategy_id, pair_id=pair_id,
            ))

    return orders
