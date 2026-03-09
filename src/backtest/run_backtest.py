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

import json

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    mlflow = None
    _HAS_MLFLOW = False

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
from backtest.engine import default_position_sizer
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


# Guide §6 — macro tickers for multivariate HMM
_MACRO_DEFAULTS = ["^VIX", "GLD", "TLT", "USO"]


def fetch_macro_features(
    macro_tickers: list,
    period: str = "15y",
    interval: str = "1d",
) -> dict:
    """Download macro assets (VIX, gold, treasuries, oil) and compute features.
    Returns {ticker: feature_df}.
    """
    client = YFinanceClient()
    macro_dict: dict = {}
    for t in macro_tickers:
        try:
            raw  = client.fetch_ticker(t, period=period, interval=interval, use_cache=True)
            feat = compute_standard_features(raw)
            feat.index = pd.to_datetime(feat["Date"]).dt.tz_localize(None)
            macro_dict[t] = feat
            logger.info("Fetched macro ticker: %s (%d rows)", t, len(feat))
        except Exception as e:
            logger.warning("Skipping macro ticker %s: %s", t, e)
    return macro_dict


def build_macro_feature_matrix(
    regime_feat_df: pd.DataFrame,
    macro_dict: dict,
) -> tuple[pd.DataFrame, list]:
    """Merge macro features into the regime-ticker feature DataFrame.

    For VIX:        long-term rolling z-score of the VIX level (already mean-reverting).
    For others:     log return, aligned to regime ticker index.

    Returns (enriched_df, list_of_new_column_names).
    """
    result     = regime_feat_df.copy()
    extra_cols: list = []

    for ticker, feat_df in macro_dict.items():
        price_col = "adj_close" if "adj_close" in feat_df.columns else "close"
        is_vix    = "VIX" in ticker.upper()

        if is_vix:
            # VIX level z-score — guide says VIX is already stationary
            vix_series = feat_df[price_col].reindex(result.index, method="ffill")
            roll_mean  = vix_series.rolling(252, min_periods=60).mean()
            roll_std   = vix_series.rolling(252, min_periods=60).std()
            col        = "vix_zscore"
            result[col] = (vix_series - roll_mean) / roll_std.replace(0, np.nan)
            extra_cols.append(col)
        else:
            # Other macro assets: log return
            col = ticker.lower().replace("^", "").replace("-", "_") + "_logret"
            if "logret" in feat_df.columns:
                result[col] = feat_df["logret"].reindex(result.index, method="ffill")
                extra_cols.append(col)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result, extra_cols


def fit_regime_detector(
    feature_dict: dict,
    regime_ticker: str,
    n_states: int = 3,
    use_walkforward: bool = True,
    min_train_years: int = 5,
    retrain_years: int = 1,
    macro_dict: dict = None,
):
    """Fit an HMMRegimeDetector following the training guide.

    Steps (guide §1–6):
      1. Feature engineering: logret, rv_20, mom_20, volume_zscore on a broad
         market proxy (VOO/SPY).
      2. Optionally enrich with macro features (VIX, GLD, TLT, USO) for a
         multivariate HMM (guide §6).
      3. Walk-forward training (use_walkforward=True): prevents look-ahead bias
         by only using data available at each prediction point.
      4. Fallback model (basic cols) stored for online inference during
         backtest simulation.
    """
    feat = feature_dict.get(regime_ticker)
    if feat is None:
        logger.warning("Regime ticker %s not found in feature_dict", regime_ticker)
        return None

    # Guide §2 — base stationary features (never train on raw prices)
    base_cols = [c for c in ["logret", "rv_20", "mom_20", "volume_zscore"] if c in feat.columns]
    extra_cols: list = []

    # Guide §6 — optional multivariate macro enrichment
    if macro_dict:
        feat, extra_cols = build_macro_feature_matrix(feat, macro_dict)
        logger.info("Macro features added: %s", extra_cols)

    all_feature_cols = base_cols + extra_cols
    det = HMMRegimeDetector(n_states=n_states, feature_cols=all_feature_cols)

    if use_walkforward:
        # Guide §3 — walk-forward training (expanding window, retrain annually)
        det.fit_predict_walkforward(
            feat,
            min_train_years=min_train_years,
            retrain_every_years=retrain_years,
        )
        logger.info(
            "Walk-forward HMM ready on %s | features=%s | %d bias-free labels",
            regime_ticker, all_feature_cols, len(det._walkforward_labels),
        )
    else:
        det.fit(feat)
        logger.info("HMM fitted (single-split) on %s | features=%s", regime_ticker, all_feature_cols)

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


def compute_regime_performance(
    equity_curve: pd.Series,
    regime_series: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """Per-regime performance breakdown (spec §4.4).

    Returns a DataFrame indexed by regime label with columns:
        Days, Ann Return %, Sharpe, Ann Vol %, Trades
    """
    import math
    from regime.base import REGIME_LABELS

    daily_rets = equity_curve.pct_change().dropna()
    regimes    = regime_series.reindex(daily_rets.index, method="ffill").dropna()
    common     = daily_rets.index.intersection(regimes.index)
    daily_rets = daily_rets[common]
    regimes    = regimes[common]

    rows = []
    for r in sorted(regimes.unique()):
        mask   = regimes == r
        r_rets = daily_rets[mask]
        if len(r_rets) < 5:
            continue

        n_days  = int(mask.sum())
        ann_ret = r_rets.mean() * 252 * 100
        ann_vol = r_rets.std() * math.sqrt(252) * 100
        daily_rf = risk_free_rate / 252
        sharpe   = (
            (r_rets.mean() - daily_rf) / r_rets.std() * math.sqrt(252)
            if r_rets.std() > 0 else float("nan")
        )

        # Count fills (rows in trades_df) that fall on dates in this regime period
        n_trades = 0
        if not trades_df.empty:
            regime_date_set = set(regimes[mask].index)
            n_trades = int(sum(1 for d in trades_df.index if d in regime_date_set))

        rows.append({
            "Regime":       REGIME_LABELS.get(int(r), f"Regime {r}"),
            "Days":         n_days,
            "Ann Return %": round(ann_ret, 2),
            "Sharpe":       round(sharpe, 2),
            "Ann Vol %":    round(ann_vol, 2),
            "Trades":       n_trades,
        })

    return pd.DataFrame(rows).set_index("Regime") if rows else pd.DataFrame()


def run_backtest(cfg: PlatformConfig, use_risk: bool = True) -> dict:
    print("=" * 64)
    print("  Regime-Adaptive Pairs Backtest")
    print("=" * 64)

    # Optionally start MLflow run to record experiment metadata and artifacts
    mlflow_run = None
    if _HAS_MLFLOW:
        try:
            exp_name = getattr(cfg.backtest, "experiment_name", "regime-adaptive-backtests")
            mlflow.set_experiment(exp_name)
            mlflow_run = mlflow.start_run()
            # Log a few key params (best-effort)
            try:
                mlflow.log_params({
                    "initial_capital": float(cfg.backtest.initial_capital),
                    "train_pct": float(cfg.backtest.train_pct),
                    "max_pairs": int(cfg.pairs.max_pairs),
                    "n_states": int(cfg.regime.n_states),
                    "use_walkforward": bool(cfg.regime.use_walkforward),
                })
            except Exception:
                pass
        except Exception:
            mlflow_run = None

    # 1. Fetch data
    print("\n[1/5] Fetching price data…")
    # Always include the regime ticker so HMM training works regardless of universe choice.
    fetch_tickers = list(cfg.data.tickers)
    regime_ticker_extra = cfg.regime.regime_ticker not in fetch_tickers
    if regime_ticker_extra:
        fetch_tickers.append(cfg.regime.regime_ticker)
    wide, feature_dict = fetch_wide_prices(
        fetch_tickers, cfg.data.period, cfg.data.interval
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

    # Fetch macro features for multivariate HMM if configured (guide §6)
    macro_dict: dict = {}
    if cfg.regime.macro_tickers:
        print(f"  Fetching macro tickers for multivariate HMM: {cfg.regime.macro_tickers}")
        macro_dict = fetch_macro_features(
            cfg.regime.macro_tickers, cfg.data.period, cfg.data.interval
        )

    regime_detector = fit_regime_detector(
        feature_dict,
        cfg.regime.regime_ticker,
        n_states=cfg.regime.n_states,
        use_walkforward=cfg.regime.use_walkforward,
        min_train_years=cfg.regime.walkforward_min_train_years,
        retrain_years=cfg.regime.walkforward_retrain_years,
        macro_dict=macro_dict if macro_dict else None,
    )

    # Collect HMM diagnostics (walk-forward label counts + transition matrix) for debugging
    hmm_debug: dict = {}
    if regime_detector is None:
        hmm_debug["error"] = f"Regime ticker '{cfg.regime.regime_ticker}' not found in fetched data"
    else:
        try:
            wf = getattr(regime_detector, "_walkforward_labels", None)
            if wf is not None and not wf.empty:
                try:
                    hmm_debug["walkforward_counts"] = {int(k): int(v) for k, v in wf.value_counts().items()}
                except Exception as e:
                    hmm_debug["walkforward_counts"] = f"error: {e}"
            else:
                hmm_debug["walkforward_counts"] = None
            try:
                tm = regime_detector.transition_matrix()
                hmm_debug["transition_matrix"] = tm.values.tolist()
                hmm_debug["n_states"] = regime_detector.n_states
            except Exception as e:
                hmm_debug["transition_matrix"] = f"error: {e}"
            # Emission means for each regime (useful for state interpretation)
            try:
                if regime_detector._model is not None:
                    order = [k for k, v in sorted(regime_detector._label_map.items(), key=lambda x: x[1])]
                    hmm_debug["emission_means"] = regime_detector._model.means_[order].tolist()
                    hmm_debug["feature_cols"] = regime_detector._active_features or list(regime_detector.feature_cols)
            except Exception as e:
                hmm_debug["emission_means"] = f"error: {e}"
        except Exception as e:
            hmm_debug["error"] = str(e)

    wide_insample = {t: v[v.index <= split_date] for t, v in
                     {t: wide[t].dropna() for t in wide.columns}.items()}
    wide_is = pd.DataFrame(wide_insample)
    # Exclude the regime ticker from pair selection (it's a market-proxy ETF, not a tradeable pair)
    if regime_ticker_extra and cfg.regime.regime_ticker in wide_is.columns:
        wide_is = wide_is.drop(columns=[cfg.regime.regime_ticker])
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
            regime_leverage_caps=getattr(cfg.risk, "regime_leverage_caps", None) or getattr(cfg.risk, "regime_leverage_caps", {}),
            regime_max_open_pairs=getattr(cfg.risk, "regime_max_open_pairs", None) or getattr(cfg.risk, "regime_max_open_pairs", {}),
            regime_pair_notional_pct=getattr(cfg.risk, "regime_pair_notional_pct", None) or getattr(cfg.risk, "regime_pair_notional_pct", {}),
            regime_ticker_notional_pct=getattr(cfg.risk, "regime_ticker_notional_pct", None) or getattr(cfg.risk, "regime_ticker_notional_pct", {}),
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
        regime_entry_z_map=cfg.pairs.regime_entry_z,
        regime_exit_z_map=cfg.pairs.regime_exit_z,
        warmup_bars=cfg.pairs.warmup_bars,
        pair_reselector=pair_reselector,
        all_tickers=cfg.data.tickers,
    )

    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        position_sizer=lambda signal, pf, prices: default_position_sizer(
            signal,
            pf,
            prices,
            target_notional_pct=cfg.backtest.target_notional_pct,
        ),
        risk_manager=risk_manager,
        verbose=cfg.backtest.verbose,
    )

    results = engine.run()
    results["selected_pairs"] = pairs_df.to_dict("records")
    results["pair_reselection_count"] = (
        pair_reselector.reselection_count if pair_reselector is not None else 0
    )

    # Always attach HMM debug info
    results["hmm_info"] = hmm_debug

    # Collect regime history for per-regime analytics (spec §4.4)
    regime_series = strategy.get_regime_history()
    results["regime_series"] = regime_series

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

    # Per-regime performance breakdown (spec §4.4)
    if not regime_series.empty:
        regime_perf = compute_regime_performance(
            equity_curve=results["equity_curve"],
            regime_series=regime_series,
            trades_df=results.get("trades", pd.DataFrame()),
        )
        results["regime_performance"] = regime_perf
        if not regime_perf.empty:
            print("\n  Regime Performance Breakdown:")
            print("-" * 48)
            print(regime_perf.to_string())

    # Plots
    _make_plots(results, pairs_df, cfg)

    # MLflow logging: metrics, artifacts, and HMM internals
    if _HAS_MLFLOW and mlflow_run is not None:
        try:
            stats = results.get("stats", {})
            # log scalar metrics
            for k, v in stats.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass

            # regime performance -> CSV artifact
            rp = results.get("regime_performance")
            if rp is not None and not rp.empty:
                rp_csv = os.path.join(cfg.plots_dir, "regime_performance.csv")
                rp.to_csv(rp_csv)
                mlflow.log_artifact(rp_csv, artifact_path="regime_performance")

            # HMM internals -> JSON artifact
            hmm_json = os.path.join(cfg.plots_dir, "hmm_info.json")
            try:
                with open(hmm_json, "w") as fh:
                    json.dump(hmm_debug, fh, indent=2)
                mlflow.log_artifact(hmm_json, artifact_path="hmm_info")
            except Exception:
                pass

            # plots
            plot_path = os.path.join(cfg.plots_dir, "backtest_results.png")
            if os.path.exists(plot_path):
                mlflow.log_artifact(plot_path, artifact_path="plots")

        except Exception:
            logger.exception("MLflow logging failed")
        finally:
            try:
                mlflow.end_run()
            except Exception:
                pass

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

    # 1. Equity curve (with regime shading — spec §4.1)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq.index, eq.values / 1e6, lw=1.2, color="steelblue", zorder=2)
    ax1.axhline(cfg.backtest.initial_capital / 1e6, ls="--", color="grey", lw=0.8)

    # Draw regime color bands in background
    regime_series = results.get("regime_series")
    if regime_series is not None and not regime_series.empty:
        _regime_colors = {0: "lightgreen", 1: "lightyellow", 2: "lightsalmon", 3: "lightcoral"}
        _regime_alphas = {0: 0.30, 1: 0.18, 2: 0.28, 3: 0.38}
        changes = regime_series.ne(regime_series.shift()).cumsum()
        for _, grp in regime_series.groupby(changes):
            r     = int(grp.iloc[0])
            color = _regime_colors.get(r, "lightgrey")
            alpha = _regime_alphas.get(r, 0.2)
            ax1.axvspan(grp.index[0], grp.index[-1],
                        facecolor=color, alpha=alpha, zorder=0)

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

    # 6. Regime performance — Sharpe by regime (spec §4.4)
    ax6 = fig.add_subplot(gs[3, 1])
    regime_perf = results.get("regime_performance")
    if regime_perf is not None and not regime_perf.empty and "Sharpe" in regime_perf.columns:
        colors = ["mediumseagreen" if v >= 0 else "tomato" for v in regime_perf["Sharpe"]]
        regime_perf["Sharpe"].plot.barh(ax=ax6, color=colors, edgecolor="white")
        ax6.axvline(0, color="black", lw=0.8)
        ax6.set_title("Sharpe Ratio by Regime")
        ax6.set_xlabel("Sharpe Ratio")
        # Annotate with trade counts
        for i, (idx, row) in enumerate(regime_perf.iterrows()):
            ax6.text(
                ax6.get_xlim()[0] + 0.02,
                i,
                f"  n={row['Trades']}",
                va="center",
                fontsize=8,
                color="black",
            )
    elif not trades.empty and "pair_id" in trades.columns:
        # Fallback: trades per pair
        pair_counts = trades["pair_id"].value_counts().head(10)
        pair_counts.plot.barh(ax=ax6, color="steelblue", edgecolor="white")
        ax6.set_title("Trades per Pair (top 10)")
        ax6.set_xlabel("# Trades")
    else:
        ax6.set_title("Regime Performance")
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
