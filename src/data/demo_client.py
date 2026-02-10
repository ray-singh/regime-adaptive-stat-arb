"""
Demo script for YFinance Client

Demonstrates fetching data for the 200 most liquid US equities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.yfinance_client import YFinanceClient
from data.universe import TOP_200_LIQUID_US_EQUITIES, get_universe, get_sector_tickers
import pandas as pd


def demo_single_ticker():
    """Demo: Fetch a single ticker."""
    print("\n" + "="*70)
    print("DEMO 1: Fetching Single Ticker (AAPL)")
    print("="*70)
    
    client = YFinanceClient(cache_dir="data/cache")
    
    # Fetch 10 years of daily data
    df = client.fetch_ticker(
        ticker="AAPL",
        period="10y",
        interval="1d"
    )
    
    print(f"\nFetched {len(df)} rows for AAPL")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nColumns: {df.columns.tolist()}")
    
    return df


def demo_multiple_tickers():
    """Demo: Fetch multiple tickers (small sample)."""
    print("\n" + "="*70)
    print("DEMO 2: Fetching Multiple Tickers (10 stocks)")
    print("="*70)
    
    client = YFinanceClient(cache_dir="data/cache")
    
    # Test with 10 tickers
    sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
                     "META", "TSLA", "JPM", "V", "UNH"]
    
    df = client.fetch_bulk(
        tickers=sample_tickers,
        period="2y",
        interval="1d",
        show_progress=True
    )
    
    print(f"\nFetched {len(df)} total rows")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nSample data:")
    print(df.head(10))
    
    # Summary
    summary = client.get_data_summary(df)
    print(f"\nData Summary:")
    print(f"  Total rows: {summary['total_rows']:,}")
    print(f"  Unique tickers: {summary['unique_tickers']}")
    print(f"  Date range: {summary['date_range']}")
    
    return df


def demo_bulk_optimized():
    """Demo: Fast bulk download using yfinance's optimized method."""
    print("\n" + "="*70)
    print("DEMO 3: Optimized Bulk Download (20 stocks)")
    print("="*70)
    
    client = YFinanceClient()
    
    # Fetch 20 tickers quickly
    sample_tickers = TOP_200_LIQUID_US_EQUITIES[:20]
    
    df = client.fetch_bulk_optimized(
        tickers=sample_tickers,
        period="1y"
    )
    
    print(f"\nFetched {len(df)} total rows")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"\nSample:")
    print(df.groupby('ticker').agg({
        'Date': ['min', 'max', 'count'],
        'close': ['min', 'max', 'mean']
    }).head(10))
    
    return df


def demo_sector_fetch():
    """Demo: Fetch tickers by sector."""
    print("\n" + "="*70)
    print("DEMO 4: Fetching by Sector (Technology)")
    print("="*70)
    
    tech_tickers = get_sector_tickers("Technology")
    print(f"Technology sector tickers: {tech_tickers}")
    
    client = YFinanceClient(cache_dir="data/cache")
    
    df = client.fetch_bulk(
        tickers=tech_tickers,
        period="1y",
        show_progress=True
    )
    
    print(f"\nFetched {len(df)} rows for {df['ticker'].nunique()} tech stocks")
    
    # Calculate daily returns
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    
    print(f"\nAverage daily returns by ticker:")
    avg_returns = df.groupby('ticker')['returns'].mean() * 100
    print(avg_returns.sort_values(ascending=False))
    
    return df


def demo_full_universe():
    """Demo: Fetch the full 200-stock universe (commented out - takes time)."""
    print("\n" + "="*70)
    print("DEMO 5: Full Universe Download (200 stocks)")
    print("="*70)
    print("\nWARNING: This will take 5-10 minutes to complete.")
    print("Uncomment the code in this function to run.\n")
    
    # Uncomment to run full download:
    # client = YFinanceClient(cache_dir="data/cache", rate_limit_delay=0.05)
    # 
    # all_tickers = get_universe("top200")
    # 
    # df = client.fetch_bulk(
    #     tickers=all_tickers,
    #     period="10y",
    #     interval="1d",
    #     show_progress=True
    # )
    # 
    # # Save to parquet
    # output_file = "data/full_universe_10y.parquet"
    # df.to_parquet(output_file, compression='snappy', index=False)
    # print(f"\nSaved full universe to {output_file}")
    # print(f"File size: {Path(output_file).stat().st_size / 1e6:.1f} MB")
    # 
    # return df
    
    return None


if __name__ == "__main__":
    print("\nYFinance Client Demo")
    print("=" * 70)
    
    # Run demos (comment out any you don't want to run)
    
    # Demo 1: Single ticker
    df1 = demo_single_ticker()
    
    # Demo 2: Multiple tickers
    df2 = demo_multiple_tickers()
    
    # Demo 3: Optimized bulk download
    df3 = demo_bulk_optimized()
    
    # Demo 4: Sector-based fetch
    df4 = demo_sector_fetch()
    
    # Demo 5: Full universe (commented out by default)
    # df5 = demo_full_universe()
    
    print("\n" + "="*70)
    print("All demos complete!")
    print("="*70)
