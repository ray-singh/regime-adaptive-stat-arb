# Regime-Adaptive Statistical Arbitrage Platform

A production-grade prototype that detects market regimes, discovers cointegrated pairs conditioned on those regimes, and provides both a regime-aware pairs trading backtest and an interactive **Regime-Aware Pair Browser** dashboard.

## Highlights

- **Data ingestion**: `yfinance` client with retry logic and native PyArrow/Parquet caching.
- **Feature pipeline**: per-ticker feature extraction (`ret`, `logret`, `rv_20`, `mom`, z-scores) + market-wide cross-asset features; snappy-compressed Parquet I/O via `pyarrow`.
- **Regime detection**: `HMMRegimeDetector` (walk-forward, multivariate with macro features: VIX, GLD, TLT, USO), `VolatilityRegimeDetector`, `ClusteringRegimeDetector`.
- **Per-regime pair discovery**: `PairDiscoveryEngine` slices price history per regime and runs correlation pre-filter → Engle–Granger cointegration → OLS hedge ratio → OU half-life (AR(1)) on each slice. Dynamic overlap threshold (`max(60, min(252, 0.6 × slice_len))`) lets discovery work on short-lived regimes.
- **Relationship analysis**: `RelationshipAnalyzer` characterises per-pair stability across regimes (stable / unstable / regime-sensitive).
- **Pair ranking**: `PairRankingEngine` scores pairs by regime sensitivity + mean-reversion strength with human-readable `stability_label` (high / medium / low).
- **Pair selection**: Engle–Granger cointegration scanner + half-life filter.
- **Periodic pair re-selection**: pairs are dynamically re-evaluated every N trading days on a trailing window — stale pairs are closed, new cointegrated pairs added.
- **Strategy**: rolling z-score pairs trading signals with entry/exit/stop rules, regime-adaptive sizing.
- **Risk manager**: pre-trade risk gatekeeper with gross/net leverage caps, per-pair and per-ticker concentration limits, max open pairs, drawdown circuit breaker (halt + reduce zones), and regime-dependent leverage caps.
- **MLflow experiment tracking**: every backtest run is recorded with a timestamped run name, full config params, per-regime performance scalars, and artifacts (equity curve Parquet, trades CSV, selected pairs CSV, HMM JSON, plots). `PairRankingEngine` logs ranking summary metrics into any active MLflow run.
- **Centralized config**: YAML / environment variable / CLI override config system.
- **Structured logging**: timestamped, leveled logging with optional file output.

## Disclaimer

- **Not investment advice:** This project is provided for educational and research purposes only. It is not investment, financial, or trading advice. It is not intended to endorse any specific trading strategy or security.
- **Risk acknowledgment:** Backtests are simplifications of real markets; past performance does not guarantee future results and this system may not produce profitable strategies in live trading.

## Project Layout

```
regime-adaptive-stat-arb/
├── src/
│   ├── config.py             # centralized PlatformConfig (YAML / env / CLI)
│   ├── pair_discovery.py     # PairDiscoveryEngine — per-regime pair discovery
│   ├── relationship_analysis.py  # RelationshipAnalyzer — stability across regimes
│   ├── pair_ranking.py       # PairRankingEngine — score + rank by interestingness
│   ├── data/                 # yfinance client (PyArrow cache), data factory, universe (200 tickers)
│   ├── features/             # featurization utilities + FeatureStore (PyArrow Parquet)
│   ├── regime/               # HMM / volatility / clustering detectors
│   ├── strategy/             # pairs trading logic + pair selector + pair re-selection
│   ├── risk/                 # risk manager (leverage, drawdown, concentration limits)
│   ├── backtest/             # event-driven backtester (execution, portfolio, engine)
│   └── ...
├── tests/                    # unit tests (pytest) for core modules
│   ├── conftest.py
│   ├── test_job_queue.py
│   ├── test_portfolio.py
│   ├── test_risk_manager.py
│   └── test_volatility_detector.py
├── dashboard/
│   ├── backend/app.py        # Flask REST API (discovery, pairs, spread endpoints)
│   └── frontend/             # React + Vite Regime-Aware Pair Browser
├── notebooks/
│   └── discovery_demo.ipynb  # interactive Plotly demo of discovery pipeline
├── config.example.yaml       # example configuration file
├── data/                     # cached OHLCV + generated plots
├── requirements.txt          # Python dependencies
└── README.md
```

## Quickstart (run from the project root)

1) Activate the project's virtualenv and install deps:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the full backtest (train/test split, in-sample pair selection, OOS backtest):

```bash
cd src

# Use a YAML config file
python -m backtest.run_backtest --config ../config.example.yaml

# Log to file
python -m backtest.run_backtest --log-level DEBUG --log-file ../logs/backtest.log
```

3) Launch the MLflow UI to inspect experiment runs:

```bash
mlflow ui --port 5002
# Open http://localhost:5002
```

Notes:
- Scripts should be run from `src/` so module imports resolve correctly.
- Cached raw OHLCV and computed features are stored under `data/` as snappy-compressed Parquet files written via the native PyArrow API.
- Copy `config.example.yaml` to `config.yaml` and customise as needed.

## Dashboard — Regime-Aware Pair Browser

An interactive pair exploration dashboard is available under `dashboard/`. It answers questions like: *Which pairs only cointegrate in bear regimes? Which spreads break down in crises?*

**Panels:**
1. **Regime Timeline** — HMM regime history with colour-coded bands (Bull / Neutral / Bear / Crisis).
2. **Top Pairs Panel** — ranked table with Score, Stability, Half-life, p-value, and Active Regimes badges per pair.
3. **Regime Comparison View** — per-regime pair lists; filter to any regime to see which pairs appear.
4. **Pair Explorer** — click any pair to open its spread time series with regime overlays and ±2σ entry/exit bands.
5. **Score / Half-life Charts** — bar chart of pair scores (colour = stability) and a half-life vs. score scatter.

Run backend:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python dashboard/backend/app.py
```

Run frontend (new terminal):

```bash
cd dashboard/frontend
npm install
npm run dev
```

Open `http://localhost:5173`, click **Run Discovery**, then explore pairs by regime.

**REST API endpoints (port 5001):**

| Endpoint | Description |
|---|---|
| `POST /api/discovery/run` | Trigger the full discovery pipeline async |
| `GET /api/discovery/status` | Poll pipeline status |
| `GET /api/pairs/ranked` | Global ranked pairs table |
| `GET /api/pairs/by-regime` | Pairs grouped by regime (with score/stability merged) |
| `GET /api/pairs/{pair_id}/spread` | Spread time series + regime overlay for a single pair |
| `GET /api/regime/series` | Regime label time series |
| `GET /api/regime/performance` | Per-regime performance breakdown |

## Architecture

### System Overview

The platform is architected as a multi-layer event-driven system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Presentation Layer                             │
│  ┌──────────────────────┐        ┌─────────────────────────────────┐    │
│  │   React Dashboard    │◄──────►│   Flask REST API                │    │
│  │   - Pair Browser     │  HTTP  │   - /discovery (async pipeline) │    │
│  │   - Regime Timeline  │        │   - /pairs/ranked               │    │
│  │   - Spread Explorer  │        │   - /pairs/by-regime            │    │
│  │   - Score Charts     │        │   - /pairs/{id}/spread          │    │
│  └──────────────────────┘        └─────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Application Layer                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  BacktestJobQueue (ThreadPoolExecutor)                           │   │
│  │  - Async job submission & tracking                               │   │
│  │  - Thread-safe with eviction policy (MAX_HISTORY=50)             │   │
│  │  - Graceful shutdown with in-flight job draining                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  PlatformConfig (Pydantic)                                       │   │
│  │  - YAML / environment / CLI layered config                       │   │
│  │  - Type-safe validation & defaults                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────-┐
│                       Backtesting Engine (Event-Driven)                  │
│                                                                          |
│  MarketEvent ──► Strategy ──► SignalEvent ──► PositionSizer ──► Order    │
│      ▲             │                                               │     │
│      │             │ (regime detection)                            │     │
│      │             │ (pair re-selection timer)                     │     │
│      │             │ (z-score signals)                             ▼     │
│      │             │                                          RiskManager│
│      │             │                                         (pre-trade) │
│      │             │                                               │     │
│      │             │                                        pass / reject│
│      │             │                                               ▼     │
│      │             │                                         OrderEvent  │
│      │             │                                               │     │
│      │             │                                               ▼     │
│  DataFeed          │                                      SimulatedBroker│
│  (iterator)        │                                      - slippage     │
│      │             │                                      - spread       │
│      │             │                                      - commission   │
│      │             │                                               │     │
│      │             │                                               ▼     │
│      │             │                                          FillEvent  │
│      │             │                                               │     │
│      │             │                                               ▼     │
│      │             └────────────────────────────────►         Portfolio  │ 
│      │                                                       (accounting)│
│      │                                                       - positions │
│      │                                                       - cash      │
│      │                                                       - equity    │
│      └───────────────────────── mark-to-market ◄──────────────────┘      │ 
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Domain Layer                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ RegimeDetect │  │  PairDiscovery   │  │  PairRankingEngine       │   │
│  │ - HMM        │  │  - Per-regime    │  │  - regime_sensitivity    │   │
│  │ - Volatility │  │    slice+test    │  │  - mean_reversion_str    │   │
│  │ - Clustering │  │  - Dynamic       │  │  - stability_label       │   │
│  └──────────────┘  │    overlap       │  └──────────────────────────┘   │
│                    └──────────────────┘                                 │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ PairSelector │  │ RelationshipAnal │  │  FeatureStore            │   │
│  │ - Coint      │  │ - stable/unstable│  │  - PyArrow Parquet cache │   │
│  │ - Half-life  │  │ - regime-sensitive  │  - rv_20, logret         │   │
│  │ - Dynamic    │  └──────────────────┘  └──────────────────────────┘   │
│  │   rescan     │                                                        │
│  └──────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │  DataClient      │  │  Universe        │  │  Cache (disk)      │     │
│  │  - yfinance      │  │  - 200 tickers   │  │  - PyArrow Parquet │     │
│  │  - Retry logic   │  │  - Sector groups │  │  - snappy compress │     │
│  │  - Rate limiting │  │                  │  │  - TTL policy      │     │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```


### Data Flow

**Discovery pipeline (dashboard):**
1. `PairDiscoveryEngine` slices the price matrix per HMM regime and runs the full statistical test suite on each slice.
2. `RelationshipAnalyzer` characterises each pair across all regimes (stable / unstable / regime-sensitive).
3. `PairRankingEngine` scores every pair and emits `score`, `stability_label`, `regime_sensitivity`, `mean_reversion_str`; if an MLflow run is active, ranking summary metrics are logged automatically.
4. Results are served by the Flask API and consumed by the React dashboard.

### Engineering Highlights

**Concurrency Safety**: `BacktestJobQueue` uses fine-grained locking to prevent race conditions. The critical fix: `submit()` atomically registers jobs and stores futures inside a single lock acquisition, eliminating a window where `cancel()` could see a job with `_future=None`.

**Resource Management**: Job queue implements eviction (keeps last 50 completed jobs) and graceful shutdown (drains in-flight jobs before terminating workers). All components support context manager protocols where applicable.

**Type Safety**: Core data structures use `@dataclass` with type hints. Config uses Pydantic for validation. All events inherit from typed base classes.

**Testability**: 87 unit tests with 100% coverage of critical paths (job queue lifecycle, portfolio accounting edge cases, risk manager rejection logic, regime detector statistics). Tests use dependency injection and mocks for external services.

**Observability**: Structured logging with configurable levels and file output. Dashboard exposes job status, equity curves, regime plots, and risk metrics in real-time.

**Configuration as Code**: YAML-based config with CLI overrides and environment variable support. Type-safe parsing with defaults and validation.

**Transaction Cost Modeling**: Realistic execution simulation with bid-ask spread, market impact slippage, and commission. Short rebate accrual on held short positions.

**Regime Adaptation**: Risk limits (leverage, max pairs, notional caps) adjust dynamically based on detected market regime. Drawdown circuit breaker halts new entries at -30%, scales down at -15%.

**Performance**: Features cached to disk via native PyArrow Parquet (snappy compression). Arrow schema embedded in every file prevents silent dtype coercion on reload. Vectorized computation with pandas/numpy. Job parallelism via ThreadPoolExecutor (GIL-friendly: numpy/statsmodels release GIL in hot paths).

### Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Backend** | Python 3.12+ | Rich quant/ML ecosystem (numpy, pandas, statsmodels) |
| **API** | Flask + flask-cors | Lightweight REST API, easy integration |
| **Job Queue** | ThreadPoolExecutor | Built-in, no external dependencies, GIL-friendly for numpy |
| **Data Source** | yfinance | Free, reliable, covers US equities |
| **Data Storage** | PyArrow + Parquet | Native Arrow API, embedded schema, snappy compression, fast I/O |
| **Experiment Tracking** | MLflow | Named runs, full param logging, per-regime metrics, artifact store |
| **Config** | YAML + Pydantic | Human-readable, type-safe validation |
| **Testing** | pytest | Rich assertion library, fixtures, parametrization |
| **Frontend** | React + Vite | Fast dev server, component architecture |
| **Charts** | Recharts | Declarative charting, React-native |
| **Regime Models** | hmmlearn + statsmodels | HMM inference + OLS/ADF econometrics |
| **Logging** | Python logging stdlib | Structured, leveled, file output |

**No External Services**: System runs entirely locally (no Redis, Postgres, message queues). MLflow defaults to a local `mlruns/` directory. Suitable for research/prototype; production would add persistent storage, message broker, and distributed task queue (Celery).

## MLflow Experiment Tracking

Every `run_backtest` call creates a named MLflow run under the `regime-adaptive-backtests` experiment (configurable via `cfg.backtest.experiment_name`).

**Logged per run:**

| Category | What is recorded |
|---|---|
| Tags | `regime_ticker`, `tickers` (first 10), `n_tickers`, `period` |
| Params | ~20 config values: capital, train_pct, max_pairs, n_states, entry/exit z, leverage caps, reselection interval, etc. |
| Metrics | All portfolio stats (Sharpe, Sortino, Calmar, max drawdown, …), `n_pairs_selected`, `pair_reselection_count`, per-regime scalars (`regime_bull_sharpe`, `regime_bear_ann_return_pct`, etc.) |
| Ranking metrics | `ranking.n_pairs`, `ranking.top_score`, `ranking.mean_score`, `ranking.mean_half_life`, `ranking.median_pvalue`, per-stability-label counts |
| Artifacts | `timeseries/equity_curve.parquet`, `trades/trades.csv`, `pairs/selected_pairs.csv`, `regime_performance/regime_performance.csv`, `hmm_info/hmm_info.json`, `plots/backtest_results.png` |

```bash
# View all runs in the browser
mlflow ui --port 5002
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

## Testing

Comprehensive unit tests cover core modules:

```bash
source .venv/bin/activate
.venv/bin/pytest tests/ -v
```

**Test Coverage** (87 tests):
- `tests/test_job_queue.py` — BacktestJobQueue stability (submit, cancel, eviction, shutdown, race conditions)
- `tests/test_portfolio.py` — Portfolio accounting (fills, positions, leverage, short rebate, performance stats)
- `tests/test_risk_manager.py` — RiskManager pre-trade checks (leverage caps, drawdown circuit breaker, regime overrides)
- `tests/test_volatility_detector.py` — VolatilityRegimeDetector (fit, predict, state stats)
