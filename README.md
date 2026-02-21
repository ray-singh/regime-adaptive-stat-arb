# Regime-Adaptive Statistical Arbitrage Platform

A sophisticated trading platform that detects market regimes, identifies statistical arbitrage opportunities, and adapts strategies based on market conditions.

## Features

- **Market Regime Detection**: Automatically identifies different market states (bull, bear, high volatility, low volatility)
- **Statistical Arbitrage**: Finds and exploits mean-reversion and cointegration opportunities
- **Adaptive Strategies**: Adjusts trading parameters based on detected regimes
- **Realistic Backtesting**: Simulates trades with realistic execution costs and slippage
- **Analytics & Visualization**: Comprehensive performance metrics and visual insights
- **Multi-Source Data**: FactSet (primary) with yfinance fallback for flexibility

## Project Structure

```
regime-adaptive-stat-arb/
├── src/
│   ├── data/              # Data fetching and management
│   │   ├── factset_client.py       # FactSet API client
│   │   ├── yfinance_client.py      # Yahoo Finance client (fallback)
│   │   ├── data_client_factory.py  # Auto-switching factory
│   │   ├── universe.py             # Stock universe definitions
│   │   └── test_*.py               # Test scripts
│   ├── regime/            # Regime detection algorithms
│   ├── strategy/          # Trading strategies
│   ├── backtest/          # Backtesting engine
│   ├── risk/              # Risk management
│   ├── analytics/         # Performance analytics
│   └── visualization/     # Charts and dashboards
├── data/                  # Cached data storage
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### 1. Clone and Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your FactSet API key:

```bash
cp .env.example .env
# Edit .env and add your FACTSET_API_KEY
```

**Note**: If you don't have a FactSet API key, the platform automatically falls back to yfinance (free).

### 3. Test Data Clients

```bash
cd src
python data/test_factset.py    # Test FactSet + factory pattern
python data/test_client.py     # Test yfinance fallback
```

## Data Sources

The platform supports multiple data sources with automatic fallback:

- **Primary**: FactSet Prices API - Institutional-grade OHLCV + adjusted close
- **Fallback**: Yahoo Finance (via yfinance) - Free historical daily OHLCV
- **Universe**: Top 200 most liquid US equities by average daily volume
- **History**: 10 years of daily data (configurable)

## Usage

### Auto-Select Data Source (Recommended)

The factory automatically uses FactSet if available, otherwise falls back to yfinance:

```python
from src.data.data_client_factory import DataClientFactory

# Auto-detect best available source
client = DataClientFactory.create(source="auto", cache_dir="data/cache")
df = client.fetch_ticker("AAPL", period="10y")
```

### FactSet Client (Explicit)

```python
from src.data.factset_client import FactSetClient

client = FactSetClient(cache_dir="data/cache")
df = client.fetch_ticker("AAPL", period="10y")

# Adjusted close is included
print(df[['Date', 'close', 'adj_close']].head())
```

### YFinance Client (Fallback)

```python
from src.data.yfinance_client import YFinanceClient

client = YFinanceClient(cache_dir="data/cache")
df = client.fetch_ticker("AAPL", period="10y")
```

### Fetch Multiple Tickers

```python
from src.data.universe import TOP_200_LIQUID_US_EQUITIES

client = DataClientFactory.create(cache_dir="data/cache")

# Fetch all 200 stocks
df = client.fetch_bulk(
    tickers=TOP_200_LIQUID_US_EQUITIES,
    period="10y",
    show_progress=True
)
```

### Get Sector-Specific Data

```python
from src.data.universe import get_sector_tickers

tech_stocks = get_sector_tickers("Technology")
df = client.fetch_bulk(tech_stocks, period="5y")
```

## Roadmap

- [x] Data ingestion from FactSet & Yahoo Finance
- [x] Stock universe definition (200 liquid US equities)
- [x] Data client factory with automatic fallback
- [ ] Regime detection algorithms (HMM, clustering, volatility-based)
- [ ] Pairs trading strategy
- [ ] Cointegration-based stat-arb
- [ ] Backtesting engine with realistic costs
- [ ] Risk management module
- [ ] Performance analytics
- [ ] Web dashboard for visualization
