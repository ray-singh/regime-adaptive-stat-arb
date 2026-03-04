# Regime-Adaptive Statistical Arbitrage Platform

A working prototype that detects market regimes, finds cointegrated pairs,
and runs a regime-aware pairs trading backtest with realistic execution costs.

## Highlights (what works today)

- Data ingestion: `yfinance` client with caching (used as the default data source).
- Feature pipeline: per-ticker feature extraction (`ret`, `logret`, `rv_20`, `mom`, z-scores).
- Regime detection: `HMMRegimeDetector`, `VolatilityRegimeDetector`, `ClusteringRegimeDetector`.
- Pair selection: Engle–Granger cointegration scanner + half-life filter.
- Strategy: rolling z-score pairs trading signals with entry/exit/stop rules.
- Backtester: event-driven engine with `Market/Signal/Order/Fill` events,
  simulated execution (half-spread + slippage + commission), portfolio accounting, and reporting.

Plots and demo scripts are included to reproduce results quickly.

## Project Layout

```
regime-adaptive-stat-arb/
├── src/
│   ├── data/          # FactSet (dev) + yfinance client, data factory, universe
│   ├── features/      # featurization utilities
│   ├── regime/        # HMM / volatility / clustering detectors
│   ├── strategy/      # pairs trading logic + pair selector
│   ├── backtest/      # event-driven backtester (execution, portfolio, engine)
│   └── ...
├── data/               # cached OHLCV + generated plots
├── requirements.txt    # Python dependencies
└── README.md
```

## Quickstart (run from the project root)

1) Activate the project's virtualenv and install deps:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the demo (feature extraction → regime detectors → pairs scan → plots):

```bash
cd src
python -m regime.demo_regime
# outputs: data/plots/regime_detection.png and data/plots/pairs_trading.png
```

3) Run the full backtest (train/test split, in-sample pair selection, OOS backtest):

```bash
cd src
python -m backtest.run_backtest
# outputs: data/plots/backtest_results.png and console performance summary
```

Notes:
- Scripts should be run from `src/` so module imports resolve correctly.
- Cached raw OHLCV and computed features are stored under `data/`.

## Current Roadmap

- [x] Data ingestion (`yfinance` fallback)
- [x] Universe definition (200 tickers)
- [x] Feature pipeline
- [x] Regime detection (HMM, volatility, clustering)
- [x] Pair selection (cointegration + half-life)
- [x] Pairs trading strategy (z-score signals)
- [x] Event-driven backtester with realistic costs
- [ ] Risk manager (position sizing, leverage limits)
- [ ] Periodic pair re-selection and productionization
- [ ] Analytics dashboard and richer visuals
- [ ] Unit tests, CI, and packaging

## Next steps (recommended)

1. Implement a risk manager that enforces max gross leverage and per-pair sizing per regime.
2. Add rolling pair re-selection (e.g., monthly/quarterly) to reduce OOS breakdowns.
3. Reduce execution friction assumptions (lower commission/slippage) when benchmarking realistic broker fees.
