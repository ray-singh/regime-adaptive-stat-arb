#!/usr/bin/env python3
"""Run the backtest for the selected best config and dump diagnostics.

Outputs to `data/analysis_best_run/`:
  - equity.csv
  - trades.csv
  - regime_performance.csv
  - summary.txt
"""
from __future__ import annotations

import os
from pathlib import Path
import json
import numpy as np

from config import PlatformConfig, setup_logging
from backtest.run_backtest import run_backtest


def main():
    cfg = PlatformConfig.from_yaml("config.example.yaml")
    setup_logging(level=cfg.log_level)

    # Best config found in grid search (picked first best):
    cfg.pairs.entry_z = 1.5
    cfg.pairs.exit_z = 0.3
    cfg.pairs.max_pairs = 5
    cfg.backtest.train_pct = 0.5

    out_dir = Path("data/analysis_best_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running backtest for diagnostics...")
    results = run_backtest(cfg, use_risk=True, use_plot=False)

    # Save equity
    eq = results.get("equity_curve")
    if eq is not None and not eq.empty:
        eq.to_csv(out_dir.joinpath("equity.csv"))

    # Save trades
    trades = results.get("trades")
    if trades is not None and not trades.empty:
        trades.to_csv(out_dir.joinpath("trades.csv"))

    # Save regime performance
    rp = results.get("regime_performance")
    if rp is not None and not rp.empty:
        rp.to_csv(out_dir.joinpath("regime_performance.csv"))

    # Basic diagnostics
    stats = results.get("stats", {})
    summary = {
        "cagr_pct": float(stats.get("cagr_pct", np.nan)),
        "sharpe_ratio": float(stats.get("sharpe_ratio", np.nan)),
        "final_equity": float(stats.get("final_equity", np.nan)),
        "n_trades": int(stats.get("n_trades", 0)),
    }

    # Trade-level diagnostics
    trade_diag = {}
    if trades is not None and not trades.empty:
        # Expect a 'pnl' column or compute from 'fill' columns if available
        if "pnl" in trades.columns:
            pnl = trades["pnl"].astype(float)
        else:
            # try to compute from 'value' or 'notional'/'side' if present
            pnl = trades.get("pnl", None)

        if pnl is not None:
            trade_diag["mean_pnl"] = float(pnl.mean())
            trade_diag["median_pnl"] = float(pnl.median())
            trade_diag["top_5_losses"] = pnl.nsmallest(5).tolist()
            trade_diag["top_5_gains"] = pnl.nlargest(5).tolist()

    # Write summary
    summary_path = out_dir.joinpath("summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Summary for best-config run\n")
        fh.write(json.dumps(summary, indent=2))
        fh.write("\n\nTrade diagnostics:\n")
        fh.write(json.dumps(trade_diag, indent=2))

    print(f"Diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
