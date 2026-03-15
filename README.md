# Regime-Adaptive Statistical Arbitrage Platform

A quantitative research system that combines unsupervised regime detection with cointegration-based pairs trading. The platform is built on the idea that relationships between trading pairs don’t stay constant. A pair that moves together in a bullish market might stop doing so when the market becomes volatile. To handle this, the system adjusts which pairs it trades, how big the positions are, and how much risk it takes based on the current market environment.

**Note**: This project is for research and educational purposes only. Nothing here constitutes investment advice.

---

## What It Does

1. **Detects market regimes** via walk-forward Hidden Markov Models trained on cross-asset features (realized volatility, return dispersion, index momentum). There are 4 regimes and they are labelled 0–3: Bull / Neutral / Bear / Crisis.
2. **Discovers cointegrated pairs per regime** — runs Engle-Granger cointegration tests on price slices belonging to each regime, computing OLS hedge ratios and Ornstein-Uhlenbeck half-lives.
3. **Ranks and filters pairs** by a composite score weighting regime sensitivity and mean-reversion speed.
4. **Backtests a pairs trading strategy** with rolling z-score signals, regime-adaptive entry/exit thresholds, and periodic pair re-selection on a trailing window.
5. **Enforces pre-trade risk rules** via a gatekeeper that checks gross/net leverage, per-pair and per-ticker concentration, maximum open pairs, and a drawdown circuit breaker — all scaled by the current regime.

---

## Statistical & ML Methods

| Component | Method | Notes |
|---|---|---|
| Regime detection | Gaussian HMM (`hmmlearn`) | Multivariate: avg_rv, ret_dispersion, index_momentum; states sorted by ascending realized vol for label stability |
| Walk-forward training | Expanding-window HMM refit | Prevents look-ahead bias; fallback model uses basic features for online inference |
| Cointegration testing | Engle-Granger (two-step OLS) | Correlation pre-filter (|r| ≥ 0.70) applied first to reduce O(n²) pair candidates |
| Hedge ratio | OLS with AR(1) residual fit | Ornstein-Uhlenbeck half-life estimated via AR(1) on spread differences |
| Dynamic hedge ratio | Kalman filter (1D, random-walk state) | Online tracking of time-varying hedge; process variance controls regime responsiveness |
| Signal generation | Rolling z-score ensemble | Combines z-score, spread momentum, and z-score-of-momentum with learned or fixed weights |
| Pair stability scoring | Cross-regime correlation std | Flags pairs whose correlation structure varies significantly across regimes |
| Unsupervised fallback | K-Means clustering | Alternative regime detector on [logret, rv_20, mom_20, z_ret_20]; states sorted by RV centroid |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  React + Recharts Dashboard  ◄──►  Flask REST API       │
│  - Regime Timeline                - /discover (async)   │
│  - Ranked Pair Browser            - /pairs/ranked       │
│  - Spread Explorer                - /pairs/{id}/spread  │
│  - Network Graph                  - /api/jobs (queue)   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  BacktestJobQueue             │
         │  ThreadPoolExecutor, max=2    │
         │  Atomic submit/cancel (lock)  │
         └───────────────┬───────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │          Event-Driven Backtest Engine    │
    │                                         │
    │  MarketEvent → Strategy → SignalEvent   │
    │      → RiskManager (pre-trade)          │
    │      → SimulatedBroker (slippage +      │
    │        spread + commission)             │
    │      → FillEvent → Portfolio            │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │              Domain Layer               │
    │  HMMRegimeDetector  PairDiscoveryEngine │
    │  RelationshipAnalyzer  PairRankingEngine│
    │  PairReSelector (periodic rescan)       │
    └────────────────────┬────────────────────┘
                         │
    ┌────────────────────▼────────────────────┐
    │               Data Layer                │
    │  yfinance client  FeatureStore          │
    │  PyArrow/Parquet cache (snappy)         │
    │  TTL-based disk cache for HMM + pairs   │
    └─────────────────────────────────────────┘
```

**Concurrency model:** A module-level `ThreadPoolExecutor` is reused across cointegration test batches (GIL is released by numpy/statsmodels in hot paths). The job queue uses fine-grained locking with atomic `submit()` to eliminate the race where `cancel()` could observe a job with `_future = None`.

**Caching:** Parquet files use the native PyArrow API with embedded Arrow schema to prevent silent dtype coercion on reload. HMM results, pair discovery, and pair ranking each have independent TTL-keyed cache files derived from MD5 hashes of their inputs.

---

## Risk Management

| Parameter | Default |
|---|---|
| Max gross leverage | 4× (Crisis regime: 1×) |
| Max net leverage | 2× |
| Per-pair notional | 20% of equity |
| Per-ticker concentration | 25% of equity |
| Max open pairs | 10 (Crisis regime: 2) |
| Drawdown halt | −30% from peak |
| Drawdown reduce | −15% (scale to 50%) |

All limits have per-regime overrides. The risk manager is a pre-trade gatekeeper called before every order reaches the simulated broker.

---

## Stack

**Python 3.12+** · hmmlearn · statsmodels · scikit-learn · pandas · numpy · PyArrow · yfinance · Flask · MLflow · pytest

**Frontend:** React 18 · Vite · Recharts

**Infrastructure:** Docker Compose · Gunicorn · nginx

---

## Testing

87 unit tests covering job queue lifecycle (including race conditions), portfolio accounting edge cases, risk manager rejection logic, and regime detector statistics. Run with:

```bash
.venv/bin/pytest tests/ -v
```

---

## Known Limitations

- **Cointegration instability.** Engle-Granger cointegration on daily equity prices is well-documented to degrade out-of-sample. Periodic re-selection partially mitigates this but does not solve structural breakdown.
- **Regime label consistency.** HMM states are sorted by realized volatility at each refit. If the volatility ordering of states changes across walk-forward windows, regime 0 in one window may not correspond to regime 0 in the next.
- **Data source.** yfinance is suitable for research but unsuitable for production — it provides adjusted closing prices with no guaranteed point-in-time correctness, survivorship-adjusted universes, or corporate action handling.
- **Execution model.** Flat-rate slippage and spread assumptions do not capture market impact at scale, intraday timing effects, or borrow cost variability for short legs.
- **Single-process state.** The Flask backend uses in-memory singletons. Gunicorn is pinned to one worker; horizontal scaling requires an external state store.

---

## Quickstart

```bash
# Backend
source .venv/bin/activate
pip install -r requirements.txt
python dashboard/backend/app.py

# Frontend
cd dashboard/frontend && npm install && npm run dev

# Full backtest
cd src && python -m backtest.run_backtest --config ../config.example.yaml

# MLflow UI
mlflow ui --port 5002
```

Or with Docker:

```bash
docker compose up --build
```

---