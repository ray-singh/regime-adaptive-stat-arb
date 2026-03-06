# Regime-Adaptive Statistical Arbitrage Platform

A production-grade prototype that detects market regimes, finds cointegrated pairs,
and runs a regime-aware pairs trading backtest with realistic execution costs,
risk management, and dynamic pair re-selection.

## Highlights

- **Data ingestion**: `yfinance` client with caching and retry logic.
- **Feature pipeline**: per-ticker feature extraction (`ret`, `logret`, `rv_20`, `mom`, z-scores).
- **Regime detection**: `HMMRegimeDetector`, `VolatilityRegimeDetector`, `ClusteringRegimeDetector`.
- **Pair selection**: Engle–Granger cointegration scanner + half-life filter.
- **Periodic pair re-selection**: pairs are dynamically re-evaluated every N trading days on a trailing window — stale pairs are closed, new cointegrated pairs added.
- **Strategy**: rolling z-score pairs trading signals with entry/exit/stop rules, regime-adaptive sizing.
- **Risk manager**: pre-trade risk gatekeeper with gross/net leverage caps, per-pair and per-ticker concentration limits, max open pairs, drawdown circuit breaker (halt + reduce zones), and regime-dependent leverage caps.
- **Backtester**: event-driven engine with `Market/Signal/Order/Fill` events,
  simulated execution (half-spread + slippage + commission), portfolio accounting, and reporting.
- **Centralized config**: YAML / environment variable / CLI override config system.
- **Structured logging**: timestamped, leveled logging with optional file output.

## Project Layout

```
regime-adaptive-stat-arb/
├── src/
│   ├── config.py      # centralized PlatformConfig (YAML / env / CLI)
│   ├── data/          # yfinance client, data factory, universe (200 tickers)
│   ├── features/      # featurization utilities + feature store
│   ├── regime/        # HMM / volatility / clustering detectors
│   ├── strategy/      # pairs trading logic + pair selector + pair re-selection
│   ├── risk/          # risk manager (leverage, drawdown, concentration limits)
│   ├── backtest/      # event-driven backtester (execution, portfolio, engine)
│   └── ...
├── config.example.yaml # example configuration file
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

4) Run with CLI options:

```bash
cd src
# Custom capital and re-selection frequency
python -m backtest.run_backtest --capital 2000000 --reselect-interval 126

# Disable risk manager
python -m backtest.run_backtest --no-risk

# Use a YAML config file
python -m backtest.run_backtest --config ../config.example.yaml

# Log to file
python -m backtest.run_backtest --log-level DEBUG --log-file ../logs/backtest.log
```

Notes:
- Scripts should be run from `src/` so module imports resolve correctly.
- Cached raw OHLCV and computed features are stored under `data/`.
- Copy `config.example.yaml` to `config.yaml` and customise as needed.

## Architecture

```
MarketEvent ──→ Strategy ──→ SignalEvent ──→ PositionSizer ──→ OrderEvent
                   │                                              │
                   │ (regime detection)                           │ (risk checks)
                   │ (pair re-selection)                          ▼
                   │                                        RiskManager
                   │                                         │  pass/reject
                   │                                         ▼
                   └─── Portfolio ◄──── FillEvent ◄──── SimulatedBroker
```

## Risk Management

The `RiskManager` (in `src/risk/`) provides:

| Check | Default |
|---|---|
| Max gross leverage | 4.0× |
| Max net leverage | 2.0× |
| Per-pair notional limit | 20% of equity |
| Per-ticker concentration | 25% of equity |
| Max open pairs | 10 |
| Drawdown halt (stop new entries) | −30% |
| Drawdown reduce (scale down) | −15% |
| Regime-dependent leverage caps | Low=4×, Normal=3×, High=2×, Crisis=1× |

## Current Roadmap

- [x] Data ingestion (`yfinance` client with caching)
- [x] Universe definition (200 tickers)
- [x] Feature pipeline
- [x] Regime detection (HMM, volatility, clustering)
- [x] Pair selection (cointegration + half-life)
- [x] Pairs trading strategy (z-score signals)
- [x] Event-driven backtester with realistic costs
- [x] Risk manager (leverage, drawdown, concentration limits)
- [x] Periodic pair re-selection
- [x] Centralized config (YAML / env / CLI)
- [x] Structured logging
- [ ] Analytics dashboard and richer visuals
- [ ] Unit tests, CI, and packaging
