#!/usr/bin/env python3
"""Quick low-turnover backtest for diagnostics.

Adjusts a few parameters to reduce trading frequency and writes results
to `data/analysis_low_turnover/`.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from config import PlatformConfig, setup_logging
from backtest.run_backtest import run_backtest


def main():
    cfg = PlatformConfig.from_yaml("config.example.yaml")
    setup_logging(level=cfg.log_level)

    # Conservative settings to cut turnover
    cfg.pairs.entry_z = 3.0
    cfg.pairs.exit_z = 1.5
    cfg.pairs.max_pairs = 3
    cfg.reselection.enabled = False
    cfg.pairs.warmup_bars = max(cfg.pairs.warmup_bars, 120)

    # Optionally shorten universe for a faster run (comment/uncomment)
    # cfg.data.tickers = cfg.data.tickers[:8]

    out_dir = Path("data/analysis_low_turnover")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running low-turnover backtest...")
    results = run_backtest(cfg, use_risk=True, use_plot=False)

    # Save summary
    stats = results.get("stats", {})
    summary = {
        "cagr_pct": float(stats.get("cagr_pct", np.nan)),
        "sharpe_ratio": float(stats.get("sharpe_ratio", np.nan)),
        "final_equity": float(stats.get("final_equity", np.nan)),
        "n_trades": int(stats.get("n_trades", 0)),
    }
    with open(out_dir.joinpath("summary.txt"), "w") as fh:
        fh.write(json.dumps(summary, indent=2))

    # Save trades and equity if available
    eq = results.get("equity_curve")
    if eq is not None and not eq.empty:
        eq.to_csv(out_dir.joinpath("equity.csv"))
    trades = results.get("trades")
    if trades is not None and not trades.empty:
        trades.to_csv(out_dir.joinpath("trades.csv"))

    rp = results.get("regime_performance")
    if rp is not None and not rp.empty:
        rp.to_csv(out_dir.joinpath("regime_performance.csv"))

    print(f"Low-turnover diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
