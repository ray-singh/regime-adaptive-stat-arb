"""Portfolio-level risk manager.

Responsibilities:
    - Pre-trade risk checks on proposed OrderEvents
    - Gross/net leverage caps
    - Per-pair notional limits
    - Total open pairs cap
    - Regime-based exposure scaling
    - Drawdown circuit breaker (reduce/halt when drawdown exceeds threshold)
    - Concentration limits (max % of equity in a single ticker)

The RiskManager sits between the PositionSizer and the Broker in the event
loop.  The engine calls `check_order()` before sending each OrderEvent to the
broker; rejected orders are silently dropped with a log entry.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from backtest.events import OrderEvent, Direction
from backtest.portfolio import Portfolio
from utils.pair_id import split_pair_id

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Tunable knobs for risk management.

    Attributes
    ----------
    max_gross_leverage : float
        Hard cap on (sum |position_notional|) / equity. Orders that would
        breach this are rejected.
    max_net_leverage : float
        Hard cap on |sum signed_position_notional| / equity.
    max_pair_notional_pct : float
        Max notional for a single pair (each leg) as % of equity.
    max_ticker_notional_pct : float
        Max notional exposure to any single ticker as % of equity.
    max_open_pairs : int
        Max number of concurrent pair positions (each pair = 2 legs).
    drawdown_halt_pct : float
        If drawdown exceeds this (negative, e.g. -0.30), all new entries
        are blocked.  Only flat/exit orders are allowed.
    drawdown_reduce_pct : float
        If drawdown exceeds this but < halt, reduce position sizes by
        `drawdown_scale_factor`.
    drawdown_scale_factor : float
        Multiplier applied to order size when in the drawdown-reduce zone.
    regime_leverage_caps : dict[int, float]
        Override max_gross_leverage per regime label.
        E.g. {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
    """
    max_gross_leverage: float = 4.0
    max_net_leverage: float = 2.0
    max_pair_notional_pct: float = 0.20     # 20% of equity per pair leg
    max_ticker_notional_pct: float = 0.25   # 25% of equity per single ticker
    max_open_pairs: int = 10
    drawdown_halt_pct: float = -0.30        # -30% drawdown → halt
    drawdown_reduce_pct: float = -0.15      # -15% drawdown → reduce
    drawdown_scale_factor: float = 0.50     # half size in reduce zone
    regime_leverage_caps: dict = field(default_factory=lambda: {
        0: 4.0,   # Low vol
        1: 3.0,   # Normal
        2: 2.0,   # High vol
        3: 1.0,   # Crisis
    })
    # Optional per-regime overrides for other risk dimensions
    regime_max_open_pairs: dict = field(default_factory=lambda: {0: 10, 1: 8, 2: 5, 3: 2})
    regime_pair_notional_pct: dict = field(default_factory=lambda: {0: 0.2, 1: 0.15, 2: 0.10, 3: 0.05})
    regime_ticker_notional_pct: dict = field(default_factory=lambda: {0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08})


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """Pre-trade risk gatekeeper.

    Usage in the engine:
        for order in orders:
            approved = risk_manager.check_order(order, portfolio, prices)
            if approved:
                fill = broker.execute(order, prices)

    Also provides `scale_order()` which can adjust order size for drawdown
    or regime risk, and `update()` for end-of-day bookkeeping.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()
        self._current_regime: int = 1     # neutral default
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._rejection_count: int = 0
        self._approval_count: int = 0
        # Track which pair_ids have open positions
        self._open_pair_ids: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────

    def check_order(
        self,
        order: OrderEvent,
        portfolio: Portfolio,
        prices: dict[str, float],
    ) -> bool:
        """Return True if the order passes all risk checks, False to reject.

        Exit/flat orders are always approved (we always want to be able to
        de-risk).
        """
        # Always allow exits
        if order.direction == Direction.FLAT:
            self._approval_count += 1
            return True

        equity = portfolio.total_equity(prices)
        if equity <= 0:
            self._reject(order, "equity <= 0")
            return False

        # 1. Drawdown circuit breaker
        if not self._check_drawdown(order, equity):
            return False

        # 2. Gross leverage
        if not self._check_gross_leverage(order, portfolio, prices, equity):
            return False

        # 3. Net leverage
        if not self._check_net_leverage(order, portfolio, prices, equity):
            return False

        # 4. Per-pair notional
        if not self._check_pair_notional(order, portfolio, prices, equity):
            return False

        # 5. Per-ticker concentration
        if not self._check_ticker_concentration(order, portfolio, prices, equity):
            return False

        # 6. Max open pairs
        if not self._check_max_pairs(order, portfolio):
            return False

        self._approval_count += 1
        return True

    def scale_order(self, order: OrderEvent, portfolio: Portfolio,
                    prices: dict[str, float]) -> OrderEvent:
        """Return a (possibly size-adjusted) order for drawdown/regime scaling.

        This is called BEFORE check_order. It can shrink the order size but
        never enlarge it. Exit orders are never scaled.
        """
        if order.direction == Direction.FLAT:
            return order

        scale = 1.0

        # Drawdown-reduce scaling
        if self._current_drawdown < self.cfg.drawdown_reduce_pct:
            scale *= self.cfg.drawdown_scale_factor

        # Regime-based leverage cap may further reduce effective target
        # (indirectly handled via gross-leverage check, but we can also
        # proactively shrink here)
        regime_cap = self.cfg.regime_leverage_caps.get(
            self._current_regime, self.cfg.max_gross_leverage
        )
        equity = portfolio.total_equity(prices)
        if equity > 0:
            current_lev = portfolio.gross_leverage(prices)
            headroom = max(0, regime_cap - current_lev)
            if headroom < 0.5:
                scale *= max(0.25, headroom / 0.5)

        if scale < 1.0 - 1e-9:
            new_qty = order.quantity * scale
            if abs(new_qty) < 0.01:
                return order  # too small to scale, let check_order reject if needed
            # Create a copy with adjusted quantity
            return OrderEvent(
                date=order.date,
                ticker=order.ticker,
                order_type=order.order_type,
                quantity=new_qty,
                direction=order.direction,
                limit_price=order.limit_price,
                strategy_id=order.strategy_id,
                pair_id=order.pair_id,
            )

        return order

    def update(self, portfolio: Portfolio, prices: dict[str, float]) -> None:
        """End-of-day bookkeeping — call once per bar after mark-to-market.

        Updates peak equity, drawdown, and open-pairs tracking.
        """
        equity = portfolio.total_equity(prices)
        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._peak_equity > 0:
            self._current_drawdown = (equity - self._peak_equity) / self._peak_equity
        else:
            self._current_drawdown = 0.0

        # Refresh open pair set from portfolio positions
        self._refresh_open_pairs(portfolio)

    def set_regime(self, regime_label: int) -> None:
        """Update the current regime label (called by the strategy or engine)."""
        self._current_regime = regime_label

    # ─────────────────────────────────────────────────────────────────────
    # Risk checks
    # ─────────────────────────────────────────────────────────────────────

    def _check_drawdown(self, order: OrderEvent, equity: float) -> bool:
        """Block new entries during halt zone."""
        if self._current_drawdown <= self.cfg.drawdown_halt_pct:
            self._reject(order, f"drawdown {self._current_drawdown:.1%} breaches "
                         f"halt threshold {self.cfg.drawdown_halt_pct:.1%}")
            return False
        return True

    def _check_gross_leverage(self, order: OrderEvent, portfolio: Portfolio,
                              prices: dict[str, float], equity: float) -> bool:
        """Reject if post-trade gross leverage would exceed cap."""
        cap = min(
            self.cfg.max_gross_leverage,
            self.cfg.regime_leverage_caps.get(self._current_regime, self.cfg.max_gross_leverage),
        )
        price = prices.get(order.ticker, 0.0)
        old_pos = portfolio.get_position(order.ticker)
        new_pos = old_pos + order.quantity

        gross_notional = sum(
            abs(shares) * prices.get(t, portfolio.avg_cost.get(t, 0.0))
            for t, shares in portfolio.positions.items()
        )
        projected_gross_notional = gross_notional - abs(old_pos) * price + abs(new_pos) * price
        post_gross = projected_gross_notional / equity if equity > 0 else float("inf")

        if post_gross > cap:
            self._reject(order, f"post-trade gross leverage {post_gross:.2f} > cap {cap:.2f}")
            return False
        return True

    def _check_net_leverage(self, order: OrderEvent, portfolio: Portfolio,
                            prices: dict[str, float], equity: float) -> bool:
        """Reject if post-trade net leverage would exceed cap."""
        price = prices.get(order.ticker, 0.0)
        old_pos = portfolio.get_position(order.ticker)
        new_pos = old_pos + order.quantity

        net_notional = sum(
            shares * prices.get(t, portfolio.avg_cost.get(t, 0.0))
            for t, shares in portfolio.positions.items()
        )
        projected_net_notional = net_notional - old_pos * price + new_pos * price
        post_net = abs(projected_net_notional / equity) if equity > 0 else float("inf")

        if post_net > self.cfg.max_net_leverage:
            self._reject(order, f"post-trade net leverage {post_net:.2f} > cap "
                         f"{self.cfg.max_net_leverage:.2f}")
            return False
        return True

    def _check_pair_notional(self, order: OrderEvent, portfolio: Portfolio,
                             prices: dict[str, float], equity: float) -> bool:
        """Reject if a single pair's notional exceeds limit."""
        if not order.pair_id:
            return True  # no pair tracking for this order

        pair_tickers = split_pair_id(order.pair_id)
        pair_notional = 0.0
        for t in pair_tickers:
            price = prices.get(t, portfolio.avg_cost.get(t, 0.0))
            shares = portfolio.get_position(t)
            if t == order.ticker:
                shares = shares + order.quantity
            pair_notional += abs(shares) * price

        post = pair_notional / equity if equity > 0 else float("inf")

        if post > self.cfg.max_pair_notional_pct:
            self._reject(order, f"pair {order.pair_id} notional {post:.1%} > "
                         f"limit {self.cfg.max_pair_notional_pct:.1%}")
            return False
        return True

    def _check_ticker_concentration(self, order: OrderEvent, portfolio: Portfolio,
                                    prices: dict[str, float], equity: float) -> bool:
        """Reject if a single ticker's total notional exceeds limit."""
        t = order.ticker
        price = prices.get(t, 0.0)
        projected_pos = portfolio.get_position(t) + order.quantity
        post = (abs(projected_pos) * price) / equity if equity > 0 else float("inf")

        if post > self.cfg.max_ticker_notional_pct:
            self._reject(order, f"ticker {t} concentration {post:.1%} > "
                         f"limit {self.cfg.max_ticker_notional_pct:.1%}")
            return False
        return True

    def _check_max_pairs(self, order: OrderEvent, portfolio: Portfolio) -> bool:
        """Reject if opening a new pair would exceed max_open_pairs."""
        if not order.pair_id:
            return True
        if order.pair_id in self._open_pair_ids:
            return True   # already open — allow adjustments
        # Use per-regime override when available
        regime_override = self.cfg.regime_max_open_pairs.get(self._current_regime, self.cfg.max_open_pairs)
        # Respect the global max_open_pairs while allowing regime to further restrict it
        max_pairs_cap = min(self.cfg.max_open_pairs, regime_override)
        if len(self._open_pair_ids) >= max_pairs_cap:
            self._reject(order, f"max open pairs ({max_pairs_cap}) reached for regime {self._current_regime}")
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _refresh_open_pairs(self, portfolio: Portfolio) -> None:
        """Rebuild the set of currently open pair IDs from portfolio positions."""
        open_tickers = set(portfolio.positions.keys())
        # A pair is "open" if at least one of its legs is held
        still_open = set()
        for pid in self._open_pair_ids:
            legs = split_pair_id(pid)
            if any(t in open_tickers for t in legs):
                still_open.add(pid)
        self._open_pair_ids = still_open

    def register_pair(self, pair_id: str) -> None:
        """Mark a pair as open (called when we first fill an order for it)."""
        self._open_pair_ids.add(pair_id)

    def _reject(self, order: OrderEvent, reason: str) -> None:
        self._rejection_count += 1
        logger.debug("RISK REJECT [%s] %s qty=%.1f: %s",
                     order.pair_id, order.ticker, order.quantity, reason)

    # ─────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return risk manager statistics."""
        total = self._approval_count + self._rejection_count
        return {
            "orders_approved": self._approval_count,
            "orders_rejected": self._rejection_count,
            "rejection_rate_pct": round(
                self._rejection_count / total * 100 if total > 0 else 0, 1
            ),
            "peak_equity": round(self._peak_equity, 2),
            "final_drawdown_pct": round(self._current_drawdown * 100, 2),
            "current_regime": self._current_regime,
            "open_pairs": len(self._open_pair_ids),
        }
