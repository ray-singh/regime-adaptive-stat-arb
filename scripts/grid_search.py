#!/usr/bin/env python3
"""Simple grid search over backtest parameters to maximize annual return.

Creates `data/grid_search_results.csv` with one row per run and prints the best
configuration by `cagr_pct`.
"""
from __future__ import annotations

import argparse
import itertools
import csv
import os
from pathlib import Path

import numpy as np

from config import PlatformConfig, setup_logging
from backtest.run_backtest import run_backtest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.example.yaml")
    p.add_argument("--out", default="data/grid_search_results.csv")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PlatformConfig.from_yaml(args.config) if args.config else PlatformConfig()
    setup_logging(level=cfg.log_level)

    # Define a modest default grid (keep runs short). Edit as needed.
    grid = {
        "entry_z": [1.5, 2.0, 2.5],
        "exit_z": [0.3, 0.5, 0.8],
        "max_pairs": [5, 10],
        "train_pct": [0.5, 0.6],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = keys + ["cagr_pct", "sharpe_ratio", "final_equity", "n_trades"]
    rows = []

    for vals in combos:
        # build config for this run
        run_cfg = PlatformConfig.from_yaml(args.config) if args.config else PlatformConfig()
        for k, v in zip(keys, vals):
            if k == "entry_z":
                run_cfg.pairs.entry_z = float(v)
            elif k == "exit_z":
                run_cfg.pairs.exit_z = float(v)
            elif k == "max_pairs":
                run_cfg.pairs.max_pairs = int(v)
            elif k == "train_pct":
                run_cfg.backtest.train_pct = float(v)

        print(f"Running grid: {dict(zip(keys, vals))}")
        try:
            results = run_backtest(run_cfg, use_risk=True, use_plot=not args.no_plot)
        except Exception as e:
            print(f"Run failed for {dict(zip(keys, vals))}: {e}")
            results = {}

        stats = results.get("stats", {}) if isinstance(results, dict) else {}
        cagr = float(stats.get("cagr_pct", float("nan"))) if stats else float("nan")
        sharpe = float(stats.get("sharpe_ratio", float("nan"))) if stats else float("nan")
        final_eq = float(stats.get("final_equity", float("nan"))) if stats else float("nan")
        n_trades = int(stats.get("n_trades", 0)) if stats else 0

        row = list(vals) + [cagr, sharpe, final_eq, n_trades]
        rows.append(row)

        # append incrementally to CSV
        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)

    # Summarize best
    if rows:
        arr = np.array(rows, dtype=float)
        cagr_vals = arr[:, len(keys)]
        best_idx = int(np.nanargmax(cagr_vals))
        best = rows[best_idx]
        print("\nGrid search complete.")
        print("Best config by CAGR:")
        print(dict(zip(keys + ["cagr_pct", "sharpe_ratio", "final_equity", "n_trades"], best)))
    else:
        print("No successful runs recorded.")


if __name__ == "__main__":
    main()
