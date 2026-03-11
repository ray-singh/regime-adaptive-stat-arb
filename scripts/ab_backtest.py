"""A/B backtest harness: baseline vs enhanced configuration.

Baseline: regime_position_scale disabled (all ones) and flat thresholds.
Enhanced: uses configured per-regime thresholds and position scaling.

Run from repo root:
    .venv_new/bin/python scripts/ab_backtest.py
"""

import json
import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from config import PlatformConfig, setup_logging
from backtest.run_backtest import run_backtest


def short_metrics(results: dict) -> dict:
    stats = results.get("stats", {})
    trades = results.get("trades", None)
    trades_count = int(len(trades)) if trades is not None else 0
    regime_perf = results.get("regime_performance")
    return {
        "final_equity": stats.get("ending_equity", None) or stats.get("final_equity", None) or None,
        "cagr_pct": stats.get("cagr_pct", None),
        "sharpe": stats.get("sharpe_ratio", None),
        "max_drawdown_pct": stats.get("max_drawdown_pct", None),
        "trades": trades_count,
        "regime_perf": regime_perf.to_dict() if regime_perf is not None else None,
    }


def run_ab():
    setup_logging(level="INFO")
    base = PlatformConfig.from_env()
    # Limit universe for quick A/B run to reduce download and compute time
    # Use a small explicit universe for a fast illustrative run
    base.data.tickers = ["VOO", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "JNJ"]
    # Shorten history and loosen pair search for a faster, illustrative run
    base.data.period = "3y"
    base.pairs.max_pairs = 6
    base.reselection.enabled = False
    base.backtest.train_pct = 0.6

    # Prepare baseline: disable per-regime sizing (flat 1.0) and flat thresholds
    baseline = deepcopy(base)
    baseline.pairs.regime_position_scale = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    baseline.pairs.regime_entry_z = {0: float(baseline.pairs.entry_z), 1: float(baseline.pairs.entry_z), 2: float(baseline.pairs.entry_z), 3: float(baseline.pairs.entry_z)}
    baseline.pairs.regime_exit_z  = {0: float(baseline.pairs.exit_z),  1: float(baseline.pairs.exit_z),  2: float(baseline.pairs.exit_z),  3: float(baseline.pairs.exit_z)}

    # Enhanced: use the platform defaults (already per-regime)
    enhanced = deepcopy(base)

    print("\nRunning baseline backtest (flat regime sizing)...\n")
    res_base = run_backtest(baseline, use_risk=True, use_plot=False)
    print("\nRunning enhanced backtest (per-regime sizing + thresholds)...\n")
    res_enh = run_backtest(enhanced, use_risk=True, use_plot=False)

    summary = {
        "baseline": short_metrics(res_base),
        "enhanced": short_metrics(res_enh),
    }

    out_path = os.path.join(ROOT, "data", "analysis_ab_test.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, default=str, indent=2)

    print("\nA/B summary saved to:", out_path)
    print(json.dumps({"baseline": {k: v for k, v in summary['baseline'].items() if k!='regime_perf'},
                      "enhanced": {k: v for k, v in summary['enhanced'].items() if k!='regime_perf'}}, indent=2, default=str))


if __name__ == '__main__':
    run_ab()
