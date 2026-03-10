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
- **Job queue stability**: thread-safe `BacktestJobQueue` with proper shutdown, race-condition fixes, and eviction logic.
- **Comprehensive testing**: 87 unit tests covering core modules (job queue, portfolio, risk manager, regime detection).

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
├── tests/             # unit tests (pytest) for core modules
│   ├── conftest.py
│   ├── test_job_queue.py
│   ├── test_portfolio.py
│   ├── test_risk_manager.py
│   └── test_volatility_detector.py
├── dashboard/         # React + Flask interactive dashboard
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

## Dashboard (React + Flask)

An interactive dashboard is available under `dashboard/`.

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

Open `http://localhost:5173`.

## Architecture

### System Overview

The platform is architected as a multi-layer event-driven system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Presentation Layer                             │
│  ┌──────────────────────┐        ┌─────────────────────────────────┐    │
│  │   React Dashboard    │◄──────►│   Flask REST API                │    │
│  │   - Real-time charts │  HTTP  │   - /backtest (async jobs)      │    │
│  │   - Risk controls    │        │   - /scenario (what-if)         │    │
│  │   - Config editor    │        │   - /config, /status            │    │
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
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │  RegimeDetector  │  │  PairSelector    │  │  FeatureStore      │     │
│  │  - HMM           │  │  - Cointegration │  │  - Caching         │     │
│  │  - Volatility    │  │  - Half-life     │  │  - Featurization   │     │
│  │  - Clustering    │  │  - Dynamic rescan│  │  - rv_20, logret   │     │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐     │
│  │  DataClient      │  │  Universe        │  │  Cache (disk)      │     │
│  │  - yfinance      │  │  - 200 tickers   │  │  - Parquet/pickle  │     │
│  │  - Retry logic   │  │  - Sector groups │  │  - TTL policy      │     │
│  │  - Rate limiting │  │                  │  │                    │     │
│  └──────────────────┘  └──────────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

**Event-Driven Architecture**: The backtest engine uses an event queue to decouple market data ingestion from strategy logic, execution, and portfolio accounting. Each component communicates via typed events (`MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`), enabling clean separation and testability.

**Dependency Injection**: Core components (strategy, broker, portfolio, risk manager) are injected into the `BacktestEngine` constructor, allowing easy mocking and unit testing.

**Strategy Pattern**: Regime detectors implement a common `BaseRegimeDetector` interface, allowing hot-swappable regime detection algorithms without changing downstream code.

**Repository Pattern**: `DataClient` abstracts data fetching with caching, retry logic, and rate limiting. Clients are swappable (yfinance, FactSet, demo) via factory pattern.

**Async Job Queue**: Dashboard operations run backtests via `BacktestJobQueue`, a thread-pool-backed async queue with job lifecycle management (pending → running → complete/failed), eviction policy, and graceful shutdown.

**Guard Pattern**: `RiskManager` acts as a pre-trade gatekeeper, rejecting orders that would violate risk limits before they reach the broker.

### Data Flow

1. **Ingestion**: `DataClient` fetches OHLCV from yfinance, caches to disk (parquet).
2. **Feature Engineering**: `FeatureStore` computes returns, realized volatility, momentum, z-scores.
3. **Regime Detection**: `RegimeDetector` fits on training data, predicts regime labels for backtest period.
4. **Pair Selection**: `PairSelector` scans for cointegrated pairs via Engle-Granger test, filters by half-life.
5. **Backtest Loop**: `BacktestEngine` iterates through bars:
   - Emits `MarketEvent` with current prices.
   - Strategy generates `SignalEvent` based on z-scores and regime.
   - PositionSizer converts signals to `OrderEvent`s (two-leg pairs).
   - `RiskManager` checks gross/net leverage, concentration, drawdown → approve/reject.
   - `SimulatedBroker` executes approved orders with slippage/commission → `FillEvent`.
   - `Portfolio` updates positions, cash, equity curve.
   - End-of-day: mark-to-market, accrue short rebate, update risk state.
6. **Periodic Re-selection**: Every N bars, strategy re-scans for new pairs, closes stale pairs.
7. **Reporting**: `Portfolio.performance_stats()` computes Sharpe, Sortino, Calmar, max drawdown, trade count.

### Engineering Highlights

**Concurrency Safety**: `BacktestJobQueue` uses fine-grained locking to prevent race conditions. The critical fix: `submit()` atomically registers jobs and stores futures inside a single lock acquisition, eliminating a window where `cancel()` could see a job with `_future=None`.

**Resource Management**: Job queue implements eviction (keeps last 50 completed jobs) and graceful shutdown (drains in-flight jobs before terminating workers). All components support context manager protocols where applicable.

**Type Safety**: Core data structures use `@dataclass` with type hints. Config uses Pydantic for validation. All events inherit from typed base classes.

**Testability**: 87 unit tests with 100% coverage of critical paths (job queue lifecycle, portfolio accounting edge cases, risk manager rejection logic, regime detector statistics). Tests use dependency injection and mocks for external services.

**Observability**: Structured logging with configurable levels and file output. Dashboard exposes job status, equity curves, regime plots, and risk metrics in real-time.

**Configuration as Code**: YAML-based config with CLI overrides and environment variable support. Type-safe parsing with defaults and validation.

**Transaction Cost Modeling**: Realistic execution simulation with bid-ask spread, market impact slippage, and commission. Short rebate accrual on held short positions.

**Regime Adaptation**: Risk limits (leverage, max pairs, notional caps) adjust dynamically based on detected market regime. Drawdown circuit breaker halts new entries at -30%, scales down at -15%.

**Performance**: Features cached to disk (parquet). Vectorized computation with pandas/numpy. Job parallelism via ThreadPoolExecutor (GIL-friendly: numpy/statsmodels release GIL in hot paths).

### Design Decisions & Trade-offs

**ThreadPoolExecutor vs ProcessPoolExecutor**: Chose threads over processes for job queue because:
- Numpy/statsmodels release GIL during computation (no serialization overhead).
- Shared memory access to price data and config (no IPC).
- Faster startup and lower memory footprint.
- Trade-off: Not CPU-bound enough to benefit from true parallelism.

**Event-Driven vs Vectorized Backtest**: Event-driven architecture chosen for:
- Realistic order-by-order execution modeling (fills depend on current portfolio state).
- Natural composition of strategy components (regime detection, risk checks, execution).
- Easy debugging (inspect queue state at any point).
- Trade-off: Slower than vectorized (batched) backtests for simple strategies.

**In-Memory Job Storage**: Jobs stored in-memory (not persisted) because:
- Dashboard use case: short-lived interactive runs, not long-running production jobs.
- Eviction policy prevents unbounded growth.
- Trade-off: Jobs lost on server restart (acceptable for prototype).

**Synchronous Risk Checks**: `RiskManager.check_order()` runs synchronously in event loop because:
- Deterministic portfolio state at decision time (no async state mutation).
- Low latency (microseconds per check).
- Trade-off: Blocks event loop, but negligible for realistic bar counts (<10k bars).

**Cointegration over Correlation**: Use cointegration for pair selection because:
- Captures long-run equilibrium relationship (stationary spread).
- Avoids spurious correlation from common trends.
- Trade-off: More compute-intensive than correlation; requires sufficient history.

**Daily Bars**: Backtest runs on daily OHLCV (not intraday) because:
- Sufficient for statistical arbitrage at research/prototype stage.
- Data availability (free via yfinance).
- Trade-off: Cannot model intraday mean reversion or execution schedules (TWAP/VWAP).

### Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Backend** | Python 3.12+ | Rich quant/ML ecosystem (numpy, pandas, statsmodels) |
| **API** | Flask + flask-cors | Lightweight REST API, easy integration |
| **Job Queue** | ThreadPoolExecutor | Built-in, no external dependencies, GIL-friendly for numpy |
| **Data Source** | yfinance | Free, reliable, covers US equities |
| **Data Storage** | Parquet (on-disk cache) | Columnar format, fast I/O, compression |
| **Config** | YAML + Pydantic | Human-readable, type-safe validation |
| **Testing** | pytest | Rich assertion library, fixtures, parametrization |
| **Frontend** | React + Vite | Fast dev server, component architecture |
| **Charts** | Recharts | Declarative charting, React-native |
| **Regime Models** | statsmodels (OLS, ADF) | Industry-standard econometrics |
| **Logging** | Python logging stdlib | Structured, leveled, file output |

**No External Services**: System runs entirely locally (no Redis, Postgres, message queues). Suitable for research/prototype; production would add persistent storage, message broker, and distributed task queue (Celery).

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
- [x] Unit tests (87 tests, 4 core modules)
- [x] Process-pool stability fixes (job queue race conditions, shutdown support)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Docker containerization
