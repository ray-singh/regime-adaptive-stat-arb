"""
Quick Start Guide: Using the Data Clients

This guide shows how to use the data clients for fetching market data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_client_factory import DataClientFactory
from data.universe import TOP_200_LIQUID_US_EQUITIES, get_sector_tickers


def example_1_auto_client():
    """Example 1: Auto-select best available data source."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Auto-Select Data Source")
    print("="*70)
    
    # Factory automatically uses FactSet if API key exists, else yfinance
    client = DataClientFactory.create(source="auto", cache_dir="data/cache")
    
    print(f"Using: {type(client).__name__}")
    
    # Fetch single ticker
    df = client.fetch_ticker("AAPL", period="1y")
    print(f"\nFetched {len(df)} rows for AAPL")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print("\nSample data:")
    print(df.head())


def example_2_explicit_source():
    """Example 2: Explicitly choose data source."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Explicit Source Selection")
    print("="*70)
    
    # Check what's available
    sources = DataClientFactory.get_available_sources()
    print(f"Available sources: {sources}")
    
    # Use yfinance explicitly (always works)
    client = DataClientFactory.create(source="yfinance", cache_dir="data/cache")
    print(f"\nUsing: {type(client).__name__}")
    
    df = client.fetch_ticker("MSFT", period="6mo")
    print(f"Fetched {len(df)} rows for MSFT")


def example_3_bulk_fetch():
    """Example 3: Fetch multiple tickers."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Bulk Ticker Fetch")
    print("="*70)
    
    client = DataClientFactory.create(cache_dir="data/cache")
    
    # Fetch 10 tech stocks
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
               "META", "TSLA", "NFLX", "ADBE", "CRM"]
    
    df = client.fetch_bulk(
        tickers=tickers,
        period="1y",
        show_progress=True
    )
    
    print(f"\nFetched data for {df['ticker'].nunique()} tickers")
    print(f"Total rows: {len(df):,}")
    print(f"\nRows per ticker:")
    print(df.groupby('ticker').size())


def example_4_sector_data():
    """Example 4: Fetch by sector."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Sector-Based Fetch")
    print("="*70)
    
    client = DataClientFactory.create(cache_dir="data/cache")
    
    # Get technology sector tickers
    tech_tickers = get_sector_tickers("Technology")
    print(f"Technology sector: {tech_tickers}")
    
    df = client.fetch_bulk(
        tickers=tech_tickers,
        period="6mo",
        show_progress=False
    )
    
    print(f"\nFetched {len(df)} rows for {df['ticker'].nunique()} tech stocks")
    
    # Calculate average daily returns by stock
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    avg_returns = df.groupby('ticker')['returns'].mean() * 100
    
    print("\nAverage daily returns (%):")
    print(avg_returns.sort_values(ascending=False))


def example_5_optimized_bulk():
    """Example 5: Fast bulk download."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Optimized Bulk Download")
    print("="*70)
    
    client = DataClientFactory.create(cache_dir="data/cache")
    
    # Fast download for 20 stocks
    tickers = TOP_200_LIQUID_US_EQUITIES[:20]
    
    print(f"Fetching {len(tickers)} tickers using optimized method...")
    df = client.fetch_bulk_optimized(
        tickers=tickers,
        period="1y"
    )
    
    print(f"\nFetched {len(df)} total rows")
    print(f"Tickers: {df['ticker'].nunique()}")
    
    # Summary stats
    summary = client.get_data_summary(df)
    print(f"\nSummary:")
    print(f"  Date range: {summary['date_range']}")
    print(f"  Avg trading days per ticker: {summary['trading_days']['mean']:.0f}")


def example_6_date_ranges():
    """Example 6: Custom date ranges."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Date Ranges")
    print("="*70)
    
    client = DataClientFactory.create(cache_dir="data/cache")
    
    # Specific date range
    df = client.fetch_ticker(
        ticker="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    print(f"Fetched AAPL for 2023")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


if __name__ == "__main__":
    print("\nData Client Quick Start Examples")
    print("="*70)
    print("\nThese examples demonstrate the data client capabilities.")
    print("Comment out examples you don't want to run.\n")
    
    # Run examples (comment out any you don't want)
    example_1_auto_client()
    example_2_explicit_source()
    example_3_bulk_fetch()
    example_4_sector_data()
    # example_5_optimized_bulk()  # Uncomment for larger dataset
    example_6_date_ranges()
    
    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Set FACTSET_API_KEY in .env to use FactSet data")
    print("2. Fetch full 200-stock universe: client.fetch_bulk(TOP_200_LIQUID_US_EQUITIES)")
    print("3. Start building regime detection features")
