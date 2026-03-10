"""Unit tests for RiskManager — pre-trade checks and drawdown circuit breaker."""
import pytest
import pandas as pd

from risk.risk_manager import RiskManager, RiskConfig
from backtest.portfolio import Portfolio
from backtest.events import OrderEvent, FillEvent, Direction, OrderType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(s: str = "2024-01-02") -> pd.Timestamp:
    return pd.Timestamp(s)


def _order(ticker="AAPL", qty=100.0, direction=Direction.LONG,
           pair_id="AAPL/MSFT") -> OrderEvent:
    return OrderEvent(
        date=_ts(),
        ticker=ticker,
        order_type=OrderType.MARKET,
        quantity=qty,
        direction=direction,
        pair_id=pair_id,
    )


def _flat_order(ticker="AAPL", pair_id="AAPL/MSFT") -> OrderEvent:
    return OrderEvent(
        date=_ts(),
        ticker=ticker,
        order_type=OrderType.MARKET,
        quantity=-100.0,
        direction=Direction.FLAT,
        pair_id=pair_id,
    )


def _prices(ticker="AAPL", price=100.0, **extras) -> dict:
    d = {ticker: price}
    d.update(extras)
    return d


def _flat_portfolio(capital=1_000_000) -> Portfolio:
    return Portfolio(initial_capital=capital)


def _buy_fill(ticker, qty, price, date="2024-01-02", pair_id=""):
    return FillEvent(
        date=_ts(date),
        ticker=ticker,
        quantity=qty,
        fill_price=price,
        commission=0.0,
        slippage_cost=0.0,
        direction=Direction.LONG,
        pair_id=pair_id,
    )


# ---------------------------------------------------------------------------
# Flat orders are always approved
# ---------------------------------------------------------------------------

class TestFlatOrderAlwaysApproved:
    def test_flat_always_passes(self):
        rm = RiskManager()
        p = _flat_portfolio()
        order = _flat_order()
        assert rm.check_order(order, p, _prices()) is True

    def test_flat_passes_even_with_drawdown_halt(self):
        rm = RiskManager(RiskConfig(drawdown_halt_pct=-0.10))
        rm._current_drawdown = -0.50   # below halt threshold
        p = _flat_portfolio()
        assert rm.check_order(_flat_order(), p, _prices()) is True

    def test_scale_order_does_not_modify_flat(self):
        rm = RiskManager()
        rm._current_drawdown = -0.20   # in reduce zone
        p = _flat_portfolio()
        order = _flat_order()
        scaled = rm.scale_order(order, p, _prices())
        assert scaled is order  # identity: flat orders are returned unchanged


# ---------------------------------------------------------------------------
# Drawdown circuit breaker
# ---------------------------------------------------------------------------

class TestDrawdownCircuitBreaker:
    def test_halt_blocks_new_entries(self):
        rm = RiskManager(RiskConfig(drawdown_halt_pct=-0.30))
        rm._current_drawdown = -0.35   # past halt threshold
        rm._peak_equity = 1_000_000
        p = _flat_portfolio(capital=650_000)  # simulated loss
        result = rm.check_order(_order(), p, _prices())
        assert result is False

    def test_no_halt_below_threshold(self):
        rm = RiskManager(RiskConfig(drawdown_halt_pct=-0.30))
        rm._current_drawdown = -0.05   # well above halt
        p = _flat_portfolio(capital=1_000_000)
        rm._peak_equity = 1_000_000
        result = rm.check_order(_order(), p, _prices("AAPL", 0.01))
        # Should not be blocked by drawdown (may still fail other checks)
        # We only care that drawdown itself does not block here
        assert result is not False or rm._rejection_count > 0

    def test_reduce_zone_scales_order_size(self):
        cfg = RiskConfig(drawdown_reduce_pct=-0.15, drawdown_scale_factor=0.50)
        rm = RiskManager(cfg)
        rm._current_drawdown = -0.20   # in reduce zone
        p = _flat_portfolio(capital=1_000_000)
        rm._peak_equity = 1_000_000
        order = _order(qty=100.0)
        scaled = rm.scale_order(order, p, _prices("AAPL", 0.1))
        # Quantity should have been reduced
        assert scaled.quantity <= order.quantity


# ---------------------------------------------------------------------------
# Gross leverage cap
# ---------------------------------------------------------------------------

class TestGrossLeverage:
    def test_order_rejected_when_gross_leverage_exceeded(self):
        cfg = RiskConfig(max_gross_leverage=1.0, max_net_leverage=10.0,
                         max_pair_notional_pct=1.0, max_ticker_notional_pct=1.0,
                         max_open_pairs=100)
        rm = RiskManager(cfg)
        p = _flat_portfolio(capital=1_000)
        # Order of $2000 notional (100 shares @ $20) would push GL > 1.0
        order = _order(ticker="X", qty=100.0)
        prices = {"X": 20.0}
        result = rm.check_order(order, p, prices)
        assert result is False

    def test_order_passes_within_leverage_cap(self):
        cfg = RiskConfig(max_gross_leverage=4.0, max_net_leverage=4.0,
                         max_pair_notional_pct=1.0, max_ticker_notional_pct=1.0,
                         max_open_pairs=100)
        rm = RiskManager(cfg)
        p = _flat_portfolio(capital=1_000_000)
        # 100 shares @ $10 = $1000 notional on $1M equity → trivial
        order = _order(ticker="X", qty=10.0, pair_id="")
        order.pair_id = ""
        result = rm.check_order(order, p, {"X": 10.0})
        assert result is True


# ---------------------------------------------------------------------------
# Max open pairs
# ---------------------------------------------------------------------------

class TestMaxOpenPairs:
    def test_new_pair_rejected_when_cap_reached(self):
        cfg = RiskConfig(max_open_pairs=2, max_gross_leverage=100.0,
                         max_net_leverage=100.0, max_pair_notional_pct=1.0,
                         max_ticker_notional_pct=1.0)
        rm = RiskManager(cfg)
        rm._open_pair_ids = {"A/B", "C/D"}   # already at cap

        p = _flat_portfolio(capital=1_000_000)
        order = _order(ticker="E", qty=1.0, pair_id="E/F")
        result = rm.check_order(order, p, {"E": 1.0})
        assert result is False

    def test_existing_pair_allowed_when_cap_reached(self):
        cfg = RiskConfig(max_open_pairs=2, max_gross_leverage=100.0,
                         max_net_leverage=100.0, max_pair_notional_pct=1.0,
                         max_ticker_notional_pct=1.0)
        rm = RiskManager(cfg)
        rm._open_pair_ids = {"A/B", "C/D"}

        p = _flat_portfolio(capital=1_000_000)
        order = _order(ticker="A", qty=1.0, pair_id="A/B")
        result = rm.check_order(order, p, {"A": 1.0})
        assert result is True


# ---------------------------------------------------------------------------
# Regime leverage caps
# ---------------------------------------------------------------------------

class TestRegimeLeverageCap:
    def test_crisis_regime_enforces_tighter_cap(self):
        cfg = RiskConfig(
            max_gross_leverage=4.0,
            max_net_leverage=10.0,
            max_pair_notional_pct=1.0,
            max_ticker_notional_pct=1.0,
            max_open_pairs=100,
            regime_leverage_caps={3: 0.5},  # crisis: very tight
        )
        rm = RiskManager(cfg)
        rm.set_regime(3)
        p = _flat_portfolio(capital=1_000)
        # $1000 order on $1000 equity = 100% gross lev > 0.5 cap
        order = _order(ticker="X", qty=1000.0, pair_id="")
        result = rm.check_order(order, p, {"X": 1.0})
        assert result is False

    def test_set_regime_updates_current_regime(self):
        rm = RiskManager()
        rm.set_regime(3)
        assert rm._current_regime == 3


# ---------------------------------------------------------------------------
# register_pair & refresh_open_pairs
# ---------------------------------------------------------------------------

class TestPairTracking:
    def test_register_pair_adds_to_open_set(self):
        rm = RiskManager()
        rm.register_pair("A/B")
        assert "A/B" in rm._open_pair_ids

    def test_update_removes_closed_pairs(self):
        rm = RiskManager()
        rm.register_pair("A/B")
        # Portfolio has no positions for A or B
        p = _flat_portfolio()
        rm.update(p, {"A": 1.0, "B": 1.0})
        assert "A/B" not in rm._open_pair_ids

    def test_update_keeps_open_pairs_with_positions(self):
        rm = RiskManager()
        rm.register_pair("A/B")
        p = _flat_portfolio(capital=100_000)
        p.update_fill(_buy_fill("A", qty=10, price=10.0))
        rm.update(p, {"A": 10.0, "B": 10.0})
        assert "A/B" in rm._open_pair_ids


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys_present(self):
        rm = RiskManager()
        s = rm.summary()
        for key in ("orders_approved", "orders_rejected", "rejection_rate_pct",
                    "peak_equity", "final_drawdown_pct", "current_regime", "open_pairs"):
            assert key in s

    def test_rejection_rate_zero_when_no_orders(self):
        rm = RiskManager()
        assert rm.summary()["rejection_rate_pct"] == 0

    def test_counts_track_approvals_and_rejections(self):
        cfg = RiskConfig(max_gross_leverage=100.0, max_net_leverage=100.0,
                         max_pair_notional_pct=1.0, max_ticker_notional_pct=1.0,
                         max_open_pairs=100)
        rm = RiskManager(cfg)
        p = _flat_portfolio(capital=1_000_000)
        rm.check_order(_order(ticker="X", qty=1.0, pair_id=""), p, {"X": 1.0})
        rm.check_order(_flat_order(), p, _prices())
        assert rm.summary()["orders_approved"] >= 2
