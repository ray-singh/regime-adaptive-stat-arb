"""Run a regime-adaptive pairs trading backtest end-to-end.

Usage (from /src):
    python -m backtest.run_backtest

Options can be edited in the CONFIG block below, or passed programmatically
via run_backtest().
"""

from __future__ import annotations

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yfinance_client import YFinanceClient
from features.featurize import compute_standard_features
from regime.hmm_detector import HMMRegimeDetector
from strategy.pairs_trading import PairsSelector
from backtest.data_feed import HistoricalDataFeed
from backtest.execution import SimulatedBroker, ExecutionConfig
from backtest.portfolio import Portfolio
from backtest.engine import BacktestEngine
from backtest.strategy_wrapper import PairsBacktestStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "JPM", "V", "UNH", "ADBE", "AMD", "AVGO", "CRM", "ORCL",
    ],
    period         = "10y",
    interval       = "1d",
    train_pct      = 0.50,           # first 50% = in-sample (fit detectors & select pairs)
    initial_capital= 1_000_000.0,
    slippage_bps   = 5.0,
    spread_bps     = 3.0,
    commission_pct = 0.001,
    zscore_window  = 60,
    entry_z        = 2.0,
    exit_z         = 0.5,
    stop_z         = 3.5,
    warmup_bars    = 60,
    max_pairs      = 10,
    regime_ticker  = "AAPL",
    plots_dir      = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "plots",
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wide_prices(tickers, period, interval) -> tuple[pd.DataFrame, dict]:
    """Returns (wide_price_df, per_ticker_feature_dict)."""
    client = YFinanceClient()
    price_dict, feature_dict = {}, {}

    for t in tickers:
        try:
            raw  = client.fetch_ticker(t, period=period, interval=interval, use_cache=True)
            feat = compute_standard_features(raw)
            feat.index = pd.to_datetime(feat["Date"]).dt.tz_localize(None)
            price_col = "adj_close" if "adj_close" in feat.columns else "close"
            price_dict[t]   = feat[price_col].rename(t)
            feature_dict[t] = feat
        except Exception as e:
            logger.warning(f"Skipping {t}: {e}")

    wide = pd.concat(price_dict.values(), axis=1).sort_index()
    return wide, feature_dict


def fit_regime_detector(feature_dict: dict, regime_ticker: str, n_states: int = 3):
    feat = feature_dict.get(regime_ticker)
    if feat is None:
        return None
    det = HMMRegimeDetector(n_states=n_states)
    det.fit(feat)
    logger.info("HMM regime detector fitted on %s", regime_ticker)
    return det


def select_pairs(wide_prices: pd.DataFrame, end_date: pd.Timestamp, max_pairs: int):
    in_sample = wide_prices[wide_prices.index <= end_date]
    selector  = PairsSelector(pvalue_threshold=0.05, min_half_life=5, max_half_life=126)
    pairs_df  = selector.find_pairs(in_sample, verbose=False, max_pairs=max_pairs)
    logger.info("Selected %d cointegrated pairs (in-sample ≤ %s)", len(pairs_df), end_date.date())
    return pairs_df


def run_backtest(config: dict = CONFIG) -> dict:
    print("=" * 64)
    print("  Regime-Adaptive Pairs Backtest")
    print("=" * 64)

    # 1. Fetch data
    print("\n[1/5] Fetching price data…")
    wide, feature_dict = fetch_wide_prices(
        config["tickers"], config["period"], config["interval"]
    )
    print(f"  Price matrix: {wide.shape[0]} days × {wide.shape[1]} tickers "
          f"({wide.index[0].date()} → {wide.index[-1].date()})")

    # 2. Train/out-of-sample split
    split_idx   = int(len(wide) * config["train_pct"])
    split_date  = wide.index[split_idx]
    print(f"\n[2/5] Train/test split at {split_date.date()} "
          f"(in-sample: {split_idx} days, out-of-sample: {len(wide)-split_idx} days)")

    # 3. In-sample: fit regime detector + select pairs
    print("\n[3/5] Fitting regime detector and selecting pairs (in-sample)…")
    regime_detector = fit_regime_detector(
        feature_dict, config["regime_ticker"]
    )

    # Trim features to in-sample for pair selection
    wide_insample = {t: v[v.index <= split_date] for t, v in
                     {t: wide[t].dropna() for t in wide.columns}.items()}
    wide_is = pd.DataFrame(wide_insample)
    pairs_df = select_pairs(wide_is, split_date, config["max_pairs"])

    if pairs_df.empty:
        print("  No cointegrated pairs found — try relaxing pvalue or half-life constraints.")
        return {}

    print(f"  Pairs found:\n{pairs_df[['ticker1','ticker2','pvalue','hedge_ratio','half_life_days']].to_string(index=False)}")

    # 4. Out-of-sample backtest
    print(f"\n[4/5] Running out-of-sample backtest (from {split_date.date()})…")
    wide_oos    = wide[wide.index >= split_date]
    feat_oos    = {t: v[v.index >= split_date] for t, v in feature_dict.items()}

    data_feed   = HistoricalDataFeed(
        price_df    = wide_oos,
        warmup_bars = config["warmup_bars"],
    )

    exec_config = ExecutionConfig(
        slippage_bps   = config["slippage_bps"],
        spread_bps     = config["spread_bps"],
        commission_pct = config["commission_pct"],
    )
    broker    = SimulatedBroker(config=exec_config)
    portfolio = Portfolio(initial_capital=config["initial_capital"])

    strategy  = PairsBacktestStrategy(
        pairs           = pairs_df.to_dict("records"),
        zscore_window   = config["zscore_window"],
        entry_z         = config["entry_z"],
        exit_z          = config["exit_z"],
        stop_z          = config["stop_z"],
        regime_detector = regime_detector,
        regime_ticker   = config["regime_ticker"],
        warmup_bars     = config["warmup_bars"],
    )

    engine = BacktestEngine(
        data_feed  = data_feed,
        strategy   = strategy,
        portfolio  = portfolio,
        broker     = broker,
        verbose    = True,
    )

    results = engine.run()

    # 5. Report
    print("\n[5/5] Performance Summary")
    print("-" * 48)
    stats = results["stats"]
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    print(f"\n  Broker costs:")
    print(f"    Total commission : ${broker.total_commission():,.2f}")
    print(f"    Total slippage   : ${broker.total_slippage():,.2f}")

    # Plots
    _make_plots(results, pairs_df, config)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _make_plots(results: dict, pairs_df: pd.DataFrame, config: dict) -> None:
    os.makedirs(config["plots_dir"], exist_ok=True)

    eq      = results.get("equity_curve", pd.Series(dtype=float))
    trades  = results.get("trades", pd.DataFrame())
    stats   = results.get("stats", {})

    if eq.empty:
        return

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values / 1e6, lw=1.2, color="steelblue")
    ax1.axhline(config["initial_capital"] / 1e6, ls="--", color="grey", lw=0.8)
    ax1.set_title(
        f"Equity Curve — Sharpe {stats.get('sharpe_ratio','?'):.2f} | "
        f"CAGR {stats.get('cagr_pct','?'):.1f}% | "
        f"MaxDD {stats.get('max_drawdown_pct','?'):.1f}%",
        fontsize=11,
    )
    ax1.set_ylabel("Equity ($M)")

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color="salmon", alpha=0.6)
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("%")

    # 3. Daily returns distribution
    ax3 = fig.add_subplot(gs[2, 0])
    daily_rets = eq.pct_change().dropna() * 100
    ax3.hist(daily_rets, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_title("Daily Return Distribution (%)")
    ax3.set_xlabel("Return (%)")

    # 4. Rolling 60-day Sharpe
    ax4 = fig.add_subplot(gs[2, 1])
    roll_sharpe = daily_rets.rolling(60).mean() / daily_rets.rolling(60).std() * (252**0.5)
    ax4.plot(roll_sharpe.index, roll_sharpe.values, lw=0.8, color="darkorange")
    ax4.axhline(0, ls="--", color="grey", lw=0.6)
    ax4.set_title("Rolling 60-Day Sharpe")

    fig.suptitle("Regime-Adaptive Pairs Trading Backtest", fontsize=14, fontweight="bold")

    out = os.path.join(config["plots_dir"], "backtest_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Backtest plot saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_backtest(CONFIG)
