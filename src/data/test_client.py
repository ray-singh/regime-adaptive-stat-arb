"""Quick test to verify yfinance client works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.yfinance_client import YFinanceClient
from data.universe import get_universe, get_sector_tickers

def test_client():
    print("Testing YFinance Client...")
    print("-" * 50)
    
    # Initialize client
    client = YFinanceClient(cache_dir="data/cache")
    print("Client initialized")
    
    # Test single ticker
    print("\n1. Testing single ticker fetch (AAPL, 6 months)...")
    df = client.fetch_ticker("AAPL", period="6mo", interval="1d")
    assert not df.empty, "No data returned for AAPL"
    assert 'ticker' in df.columns, "Missing ticker column"
    assert 'close' in df.columns, "Missing close column"
    print(f"✓ Fetched {len(df)} rows for AAPL")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Test multiple tickers
    print("\n2. Testing bulk fetch (3 tickers, 3 months)...")
    test_tickers = ["MSFT", "GOOGL", "AMZN"]
    df_bulk = client.fetch_bulk(
        tickers=test_tickers,
        period="3mo",
        show_progress=False
    )
    assert not df_bulk.empty, "No data returned for bulk fetch"
    assert df_bulk['ticker'].nunique() == len(test_tickers), "Missing tickers in result"
    print(f"✓ Fetched {len(df_bulk)} rows for {df_bulk['ticker'].nunique()} tickers")
    
    # Test universe access
    print("\n3. Testing universe definitions...")
    universe = get_universe("top200")
    assert len(universe) == 200, f"Expected 200 tickers, got {len(universe)}"
    print(f"✓ Universe has {len(universe)} tickers")
    print(f"  Sample: {universe[:5]}")
    
    # Test sector access
    print("\n4. Testing sector definitions...")
    tech_tickers = get_sector_tickers("Technology")
    assert len(tech_tickers) > 0, "No tech tickers found"
    print(f"✓ Technology sector has {len(tech_tickers)} tickers")
    print(f"  Tickers: {tech_tickers}")
    
    # Test data summary
    print("\n5. Testing data summary...")
    summary = client.get_data_summary(df_bulk)
    print(f"✓ Summary generated:")
    print(f"  Total rows: {summary['total_rows']:,}")
    print(f"  Unique tickers: {summary['unique_tickers']}")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED!")
    print("="*50)
    print("\nYFinance client is working correctly.")
    print("Ready to fetch data for the full 200-stock universe.")

if __name__ == "__main__":
    try:
        test_client()
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
