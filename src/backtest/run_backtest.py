"""Run a regime-adaptive pairs trading backtest end-to-end.

Usage (from src/):
    python -m backtest.run_backtest                     # defaults
    python -m backtest.run_backtest --config ../config.yaml
    python -m backtest.run_backtest --capital 2000000 --train-pct 0.60

Features:
    - Centralized PlatformConfig (YAML / env-var / CLI overrides)
    - Risk manager with leverage caps, drawdown circuit breaker
    - Periodic pair re-selection (every N trading days)
    - Structured logging
"""

from __future__ import annotations

import argparse
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

from config import PlatformConfig, setup_logging
from data.yfinance_client import YFinanceClient
from features.featurize import compute_standard_features
from regime.hmm_detector import HMMRegimeDetector
from strategy.pairs_trading import PairsSelector
from strategy.pair_reselection import PairReSelector
from risk.risk_manager import RiskManager, RiskConfig
from backtest.data_feed import HistoricalDataFeed
from backtest.execution import SimulatedBroker, ExecutionConfig
from backtest.portfolio import Portfolio
from backtest.engine import BacktestEngine
from backtest.strategy_wrapper import PairsBacktestStrategy

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regime-Adaptive Statistical Arbitrage Backtest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file")
    p.add_argument("--capital", type=float, default=None,
                   help="Initial capital ($)")
    p.add_argument("--train-pct", type=float, default=None,
                   help="Fraction of data for in-sample training (0-1)")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Max simultaneous pairs")
    p.add_argument("--reselect-interval", type=int, default=None,
                   help="Pair re-selection interval (trading days)")
    p.add_argument("--no-reselect", action="store_true",
                   help="Disable periodic pair re-selection")
    p.add_argument("--no-risk", action="store_true",
                   help="Disable risk manager")
    p.add_argument("--log-level", type=str, default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-file", type=str, default=None,
                   help="Also write logs to this file")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PlatformConfig:
    """Construct PlatformConfig from YAML → env → CLI overrides."""
    if args.config:
        cfg = PlatformConfig.from_yaml(args.config)
    else:
        cfg = PlatformConfig.from_env()

    # CLI overrides
    if args.capital is not None:
        cfg.backtest.initial_capital = args.capital
    if args.train_pct is not None:
        cfg.backtest.train_pct = args.train_pct
    if args.max_pairs is not None:
        cfg.pairs.max_pairs = args.max_pairs
    if args.reselect_interval is not None:
        cfg.reselection.interval_days = args.reselect_interval
    if args.no_reselect:
        cfg.reselection.enabled = False
    if args.log_level:
        cfg.log_level = args.log_level

    return cfg


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
            logger.warning("Skipping %s: %s", t, e)

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


def select_pairs(wide_prices: pd.DataFrame, end_date: pd.Timestamp,
                 cfg: PlatformConfig) -> pd.DataFrame:
    in_sample = wide_prices[wide_prices.index <= end_date]
    selector  = PairsSelector(
        pvalue_threshold=cfg.pairs.pvalue_threshold,
        min_half_life=cfg.pairs.min_half_life,
        max_half_life=cfg.pairs.max_half_life,
    )
    pairs_df  = selector.find_pairs(in_sample, verbose=False,
                                    max_pairs=cfg.pairs.max_pairs)
    logger.info("Selected %d cointegrated pairs (in-sample ≤ %s)",
                len(pairs_df), end_date.date())
    return pairs_df


def run_backtest(cfg: PlatformConfig, use_risk: bool = True) -> dict:
    print("=" * 64)
    print("  Regime-Adaptive Pairs Backtest")
    print("=" * 64)

    # 1. Fetch data
    print("\n[1/5] Fetching price data…")
    wide, feature_dict = fetch_wide_prices(
        cfg.data.tickers, cfg.data.period, cfg.data.interval
    )
    print(f"  Price matrix: {wide.shape[0]} days × {wide.shape[1]} tickers "
          f"({wide.index[0].date()} → {wide.index[-1].date()})")

    # 2. Train/out-of-sample split
    split_idx   = int(len(wide) * cfg.backtest.train_pct)
    split_date  = wide.index[split_idx]
    print(f"\n[2/5] Train/test split at {split_date.date()} "
          f"(in-sample: {split_idx} days, out-of-sample: {len(wide)-split_idx} days)")

    # 3. In-sample: fit regime detector + select pairs
    print("\n[3/5] Fitting regime detector and selecting pairs (in-sample)…")
    regime_detector = fit_regime_detector(
        feature_dict, cfg.regime.regime_ticker, cfg.regime.n_states
    )

    wide_insample = {t: v[v.index <= split_date] for t, v in
                     {t: wide[t].dropna() for t in wide.columns}.items()}
    wide_is = pd.DataFrame(wide_insample)
    pairs_df = select_pairs(wide_is, split_date, cfg)

    if pairs_df.empty:
        print("  No cointegrated pairs found — try relaxing pvalue or half-life constraints.")
        return {}

    print(f"  Pairs found:\n{pairs_df[['ticker1','ticker2','pvalue','hedge_ratio','half_life_days']].to_string(index=False)}")

    # 4. Out-of-sample backtest
    print(f"\n[4/5] Running out-of-sample backtest (from {split_date.date()})…")
    wide_oos = wide[wide.index >= split_date]

    data_feed = HistoricalDataFeed(
        price_df=wide_oos,
        warmup_bars=cfg.pairs.warmup_bars,
    )

    exec_config = ExecutionConfig(
        slippage_bps=cfg.execution.slippage_bps,
        spread_bps=cfg.execution.spread_bps,
        commission_pct=cfg.execution.commission_pct,
        min_commission=cfg.execution.min_commission,
    )
    broker    = SimulatedBroker(config=exec_config)
    portfolio = Portfolio(initial_capital=cfg.backtest.initial_capital)

    # Risk manager
    risk_manager = None
    if use_risk:
        risk_cfg = RiskConfig(
            max_gross_leverage=cfg.risk.max_gross_leverage,
            max_net_leverage=cfg.risk.max_net_leverage,
            max_pair_notional_pct=cfg.risk.max_pair_notional_pct,
            max_ticker_notional_pct=cfg.risk.max_ticker_notional_pct,
            max_open_pairs=cfg.risk.max_open_pairs,
            drawdown_halt_pct=cfg.risk.drawdown_halt_pct,
            drawdown_reduce_pct=cfg.risk.drawdown_reduce_pct,
            drawdown_scale_factor=cfg.risk.drawdown_scale_factor,
        )
        risk_manager = RiskManager(config=risk_cfg)
        risk_manager._peak_equity = cfg.backtest.initial_capital
        logger.info("Risk manager enabled: %s", risk_cfg)

    # Pair re-selector
    pair_reselector = None
    if cfg.reselection.enabled:
        pair_reselector = PairReSelector(
            reselection_interval=cfg.reselection.interval_days,
            lookback_days=cfg.reselection.lookback_days,
            pvalue_threshold=cfg.pairs.pvalue_threshold,
            min_half_life=cfg.pairs.min_half_life,
            max_half_life=cfg.pairs.max_half_life,
            max_pairs=cfg.pairs.max_pairs,
        )
        logger.info("Pair re-selection enabled every %d days", cfg.reselection.interval_days)

    strategy = PairsBacktestStrategy(
        pairs=pairs_df.to_dict("records"),
        zscore_window=cfg.pairs.zscore_window,
        entry_z=cfg.pairs.entry_z,
        exit_z=cfg.pairs.exit_z,
        stop_z=cfg.pairs.stop_z,
        regime_detector=regime_detector,
        regime_ticker=cfg.regime.regime_ticker,
        warmup_bars=cfg.pairs.warmup_bars,
        pair_reselector=pair_reselector,
        all_tickers=cfg.data.tickers,
    )

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        risk_manager=risk_manager,
        verbose=cfg.backtest.verbose,
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

    if "risk_summary" in results:
        print(f"\n  Risk Manager:")
        for k, v in results["risk_summary"].items():
            print(f"    {k:30s}: {v}")

    if pair_reselector:
        print(f"\n  Pair Re-selections: {pair_reselector.reselection_count}")

    # Plots
    _make_plots(results, pairs_df, cfg)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _make_plots(results: dict, pairs_df: pd.DataFrame, cfg: PlatformConfig) -> None:
    os.makedirs(cfg.plots_dir, exist_ok=True)

    eq      = results.get("equity_curve", pd.Series(dtype=float))
    trades  = results.get("trades", pd.DataFrame())
    stats   = results.get("stats", {})

    if eq.empty:
        return

    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.3)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values / 1e6, lw=1.2, color="steelblue")
    ax1.axhline(cfg.backtest.initial_capital / 1e6, ls="--", color="grey", lw=0.8)
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

    # 5. Risk manager summary
    ax5 = fig.add_subplot(gs[3, 0])
    if "risk_summary" in results:
        ax5.text(0.5, 0.5,
                 f"Approved: {results['risk_summary']['orders_approved']}\n"
                 f"Rejected: {results['risk_summary']['orders_rejected']}\n"
                 f"Rejection rate: {results['risk_summary']['rejection_rate_pct']}%\n"
                 f"Peak equity: ${results['risk_summary']['peak_equity']:,.0f}\n"
                 f"Final DD: {results['risk_summary']['final_drawdown_pct']:.1f}%",
                 transform=ax5.transAxes, fontsize=10, verticalalignment='center',
                 horizontalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax5.set_title("Risk Manager Summary")
    else:
        ax5.set_title("Risk Manager (disabled)")
    ax5.set_xticks([])
    ax5.set_yticks([])

    # 6. Trade count by pair
    ax6 = fig.add_subplot(gs[3, 1])
    if not trades.empty and "pair_id" in trades.columns:
        pair_counts = trades["pair_id"].value_counts().head(10)
        pair_counts.plot.barh(ax=ax6, color="steelblue", edgecolor="white")
        ax6.set_title("Trades per Pair (top 10)")
        ax6.set_xlabel("# Trades")
    else:
        ax6.set_title("Trades per Pair")
        ax6.set_xticks([])
        ax6.set_yticks([])

    fig.suptitle("Regime-Adaptive Pairs Trading Backtest", fontsize=14, fontweight="bold")

    out = os.path.join(cfg.plots_dir, "backtest_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Backtest plot saved → {out}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = build_config(args)
    setup_logging(level=cfg.log_level, log_file=args.log_file)
    use_risk = not args.no_risk
    run_backtest(cfg, use_risk=use_risk)


if __name__ == "__main__":
    main()
