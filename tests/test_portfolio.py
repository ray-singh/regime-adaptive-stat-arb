"""Unit tests for Portfolio accounting."""
import math

import pandas as pd
import pytest

from backtest.portfolio import Portfolio
from backtest.events import FillEvent, Direction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _buy_fill(ticker, qty, price, commission=1.0, date="2024-01-02", pair_id=""):
    """Helper: create a buy FillEvent."""
    total = -(qty * price) - commission
    return FillEvent(
        date=_ts(date),
        ticker=ticker,
        quantity=qty,
        fill_price=price,
        commission=commission,
        slippage_cost=0.0,
        direction=Direction.LONG,
        pair_id=pair_id,
    )


def _sell_fill(ticker, qty, price, commission=1.0, date="2024-01-03", pair_id=""):
    """Helper: create a sell (short) FillEvent. qty is negative."""
    return FillEvent(
        date=_ts(date),
        ticker=ticker,
        quantity=qty,           # negative
        fill_price=price,
        commission=commission,
        slippage_cost=0.0,
        direction=Direction.SHORT,
        pair_id=pair_id,
    )


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_cash_equals_initial_capital(self):
        p = Portfolio(initial_capital=500_000)
        assert p.cash == 500_000

    def test_no_positions(self):
        p = Portfolio()
        assert p.positions == {}

    def test_total_equity_equals_cash_when_flat(self):
        p = Portfolio(initial_capital=1_000_000)
        assert p.total_equity({}) == 1_000_000


# ---------------------------------------------------------------------------
# update_fill — Long
# ---------------------------------------------------------------------------

class TestUpdateFillLong:
    def test_buy_increases_position(self):
        p = Portfolio(initial_capital=100_000)
        fill = _buy_fill("AAPL", qty=100, price=150.0, commission=1.0)
        p.update_fill(fill)
        assert p.positions["AAPL"] == 100
        expected_cash = 100_000 - (100 * 150.0) - 1.0
        assert abs(p.cash - expected_cash) < 1e-6

    def test_avg_cost_on_new_long(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_buy_fill("AAPL", qty=100, price=150.0))
        assert abs(p.avg_cost["AAPL"] - 150.0) < 1e-6

    def test_avg_cost_weighted_on_add_to_long(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_buy_fill("AAPL", qty=100, price=100.0, commission=0))
        p.update_fill(_buy_fill("AAPL", qty=100, price=200.0, commission=0, date="2024-01-03"))
        # avg should be 150
        assert abs(p.avg_cost["AAPL"] - 150.0) < 1e-6
        assert p.positions["AAPL"] == 200

    def test_sell_reduces_long_position(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_buy_fill("AAPL", qty=100, price=150.0, commission=0))
        p.update_fill(_sell_fill("AAPL", qty=-50, price=160.0, commission=0))
        assert p.positions["AAPL"] == 50

    def test_position_removed_when_flat(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_buy_fill("AAPL", qty=100, price=150.0, commission=0))
        p.update_fill(_sell_fill("AAPL", qty=-100, price=160.0, commission=0))
        assert "AAPL" not in p.positions

    def test_trade_log_populated(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_buy_fill("AAPL", qty=10, price=100.0, commission=1.0))
        assert len(p._trades) == 1
        assert p._trades[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# update_fill — Short
# ---------------------------------------------------------------------------

class TestUpdateFillShort:
    def test_short_sale_creates_negative_position(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_sell_fill("TSLA", qty=-50, price=200.0, commission=0))
        assert p.positions["TSLA"] == -50

    def test_short_sale_proceeds_tracked(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_sell_fill("TSLA", qty=-50, price=200.0, commission=0))
        # Short proceeds = 50 * 200 = 10,000
        assert abs(p.short_proceeds.get("TSLA", 0) - 10_000) < 1e-6

    def test_cover_short_removes_proceeds(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_sell_fill("TSLA", qty=-100, price=200.0, commission=0))
        p.update_fill(_buy_fill("TSLA", qty=100, price=190.0, commission=0, date="2024-01-04"))
        assert "TSLA" not in p.positions
        # Short proceeds should be fully released
        assert p.short_proceeds.get("TSLA", 0.0) < 1e-6

    def test_partial_cover_reduces_proceeds_proportionally(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_sell_fill("TSLA", qty=-100, price=200.0, commission=0))
        proceeds_before = p.short_proceeds["TSLA"]
        p.update_fill(_buy_fill("TSLA", qty=50, price=190.0, commission=0, date="2024-01-04"))
        expected = proceeds_before * 0.5
        assert abs(p.short_proceeds.get("TSLA", 0) - expected) < 1e-6


# ---------------------------------------------------------------------------
# mark_to_market & total_equity
# ---------------------------------------------------------------------------

class TestMarkToMarket:
    def test_mark_to_market_returns_equity(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_buy_fill("AAPL", qty=100, price=100.0, commission=0))
        equity = p.mark_to_market(_ts("2024-01-02"), {"AAPL": 110.0})
        # cash = 100_000 - 10_000 = 90_000; position MV = 100 * 110 = 11_000
        assert abs(equity - 101_000) < 1e-6

    def test_equity_curve_grows_after_mark(self):
        p = Portfolio(initial_capital=100_000)
        p.mark_to_market(_ts("2024-01-02"), {})
        p.mark_to_market(_ts("2024-01-03"), {})
        assert len(p._equity_curve) == 2

    def test_total_equity_long_plus_cash(self):
        p = Portfolio(initial_capital=1_000)
        p.update_fill(_buy_fill("X", qty=10, price=10.0, commission=0))
        # cash = 900, position = 10 shares @ 12 = 120
        assert abs(p.total_equity({"X": 12.0}) - 1_020) < 1e-6

    def test_total_equity_short_reduces_equity(self):
        p = Portfolio(initial_capital=10_000)
        # Short 100 shares @ 50: receive +5000 cash, position = -100
        p.update_fill(_sell_fill("X", qty=-100, price=50.0, commission=0))
        # cash = 10000 + 5000 = 15000; position MV = -100 * 60 = -6000
        assert abs(p.total_equity({"X": 60.0}) - 9_000) < 1e-6

    def test_falls_back_to_avg_cost_when_price_missing(self):
        p = Portfolio(initial_capital=10_000)
        p.update_fill(_buy_fill("X", qty=10, price=50.0, commission=0))
        # No price provided — should use avg_cost=50
        equity = p.total_equity({})
        assert abs(equity - 10_000) < 1e-6   # cost basis = 50 * 10 = 500; cash = 9500


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------

class TestLeverage:
    def test_gross_leverage_zero_when_flat(self):
        p = Portfolio(initial_capital=100_000)
        assert p.gross_leverage({}) == 0.0

    def test_gross_leverage_with_long(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_buy_fill("X", qty=100, price=100.0, commission=0))
        # cash = 90_000; equity = 100_000 at cost; gross notional = 10_000
        lev = p.gross_leverage({"X": 100.0})
        assert abs(lev - 0.1) < 1e-6

    def test_net_leverage_long_short_offsets(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_buy_fill("A", qty=100, price=100.0, commission=0))
        p.update_fill(_sell_fill("B", qty=-100, price=100.0, commission=0))
        # Net notional ≈ 0
        net = p.net_leverage({"A": 100.0, "B": 100.0})
        assert abs(net) < 1e-4

    def test_margin_available_reduced_by_short(self):
        p = Portfolio(initial_capital=200_000)
        p.update_fill(_sell_fill("X", qty=-100, price=100.0, commission=0))
        # Short proceeds = 10_000; pledged = 0.5 * 10_000 = 5_000
        # cash = 200_000 + 10_000 = 210_000 (sell proceeds added to cash)
        pledged = p.margin_requirement * 10_000
        expected = p.cash - pledged
        assert abs(p.margin_available() - expected) < 1e-6


# ---------------------------------------------------------------------------
# Short rebate accrual
# ---------------------------------------------------------------------------

class TestShortRebate:
    def test_rebate_increases_cash(self):
        p = Portfolio(initial_capital=100_000)
        p.update_fill(_sell_fill("X", qty=-100, price=100.0, commission=0))
        cash_before = p.cash
        p.accrue_short_rebate(rebate_rate_annual=0.04)
        assert p.cash > cash_before

    def test_rebate_zero_when_flat(self):
        p = Portfolio(initial_capital=100_000)
        cash_before = p.cash
        p.accrue_short_rebate(rebate_rate_annual=0.04)
        assert p.cash == cash_before


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------

class TestPerformanceStats:
    def _build_equity_curve(self, p, values):
        """Inject a synthetic equity curve."""
        dates = pd.date_range("2020-01-02", periods=len(values), freq="B")
        p._equity_curve = list(zip(dates, values))

    def test_empty_returns_empty_dict(self):
        p = Portfolio()
        assert p.performance_stats() == {}

    def test_single_point_returns_empty(self):
        p = Portfolio()
        p._equity_curve = [(_ts("2020-01-02"), 1_000_000)]
        assert p.performance_stats() == {}

    def test_positive_return_gives_positive_cagr(self):
        p = Portfolio(initial_capital=1_000_000)
        # ~1 year of steady growth: 252 daily points from 1M to ~1.1M
        n = 252
        values = [1_000_000 * (1.1 ** (i / n)) for i in range(n)]
        self._build_equity_curve(p, values)
        stats = p.performance_stats()
        assert stats["cagr_pct"] > 0

    def test_sharpe_defined_for_non_constant_series(self):
        import numpy as np
        p = Portfolio(initial_capital=1_000_000)
        rng = np.random.default_rng(42)
        rets = rng.normal(0.0003, 0.01, 252)
        values = [1_000_000]
        for r in rets:
            values.append(values[-1] * (1 + r))
        self._build_equity_curve(p, values)
        stats = p.performance_stats()
        assert not math.isnan(stats["sharpe_ratio"])

    def test_max_drawdown_is_non_positive(self):
        p = Portfolio(initial_capital=1_000_000)
        # Goes up then down
        values = [1_000_000, 1_100_000, 1_050_000, 900_000, 950_000]
        self._build_equity_curve(p, values)
        stats = p.performance_stats()
        assert stats["max_drawdown_pct"] <= 0
