"""Test script for FactSet client."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.factset_client import FactSetClient
from data.data_client_factory import DataClientFactory
from data.universe import get_universe


def test_factset_client():
    """Test FactSet client functionality."""
    print("\n" + "="*70)
    print("FACTSET CLIENT TEST")
    print("="*70)
    
    # Check API key
    if not os.getenv("FACTSET_API_KEY"):
        print("❌ FACTSET_API_KEY not found in environment")
        print("Set the environment variable to run FactSet tests")
        return False
    
    print("✓ FACTSET_API_KEY found")
    
    try:
        # Initialize client
        print("\n1. Initializing FactSet client...")
        client = FactSetClient(cache_dir="data/cache_factset")
        print("✓ Client initialized")
        
        # Test single ticker
        print("\n2. Testing single ticker fetch (AAPL, 6 months)...")
        df = client.fetch_ticker("AAPL", period="6mo")
        
        if df.empty:
            print("❌ No data returned for AAPL")
            return False
        
        print(f"✓ Fetched {len(df)} rows for AAPL")
        print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\n  Sample data:")
        print(df.head(3))
        
        # Verify adjusted close
        if 'adj_close' in df.columns:
            print("✓ Adjusted close column present")
        else:
            print("⚠ Warning: adj_close column missing")
        
        # Test multiple tickers
        print("\n3. Testing bulk fetch (3 tickers, 3 months)...")
        test_tickers = ["MSFT", "GOOGL", "AMZN"]
        df_bulk = client.fetch_bulk(
            tickers=test_tickers,
            period="3mo",
            show_progress=False
        )
        
        if df_bulk.empty:
            print("❌ No data returned for bulk fetch")
            return False
        
        print(f"✓ Fetched {len(df_bulk)} rows for {df_bulk['ticker'].nunique()} tickers")
        print(f"  Tickers present: {df_bulk['ticker'].unique().tolist()}")
        
        # Test data summary
        print("\n4. Testing data summary...")
        summary = client.get_data_summary(df_bulk)
        print(f"✓ Summary generated:")
        print(f"  Total rows: {summary['total_rows']:,}")
        print(f"  Unique tickers: {summary['unique_tickers']}")
        
        print("\n" + "="*70)
        print("✅ ALL FACTSET TESTS PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_factory():
    """Test data client factory with auto fallback."""
    print("\n" + "="*70)
    print("DATA CLIENT FACTORY TEST")
    print("="*70)
    
    # Check available sources
    print("\n1. Checking available data sources...")
    sources = DataClientFactory.get_available_sources()
    print(f"✓ Available sources: {sources}")
    
    # Test auto-selection
    print("\n2. Testing auto-selection (with fallback)...")
    client = DataClientFactory.create(source="auto", cache_dir="data/cache")
    print(f"✓ Client created: {type(client).__name__}")
    
    # Test fetch
    print("\n3. Testing data fetch with auto-selected client...")
    df = client.fetch_ticker("AAPL", period="1mo")
    
    if df.empty:
        print("❌ No data returned")
        return False
    
    print(f"✓ Fetched {len(df)} rows for AAPL")
    print(f"  Data source: {type(client).__name__}")
    
    # Test explicit source selection
    if DataClientFactory.is_source_available("factset"):
        print("\n4. Testing explicit FactSet selection...")
        factset_client = DataClientFactory.create(source="factset", cache_dir="data/cache")
        print(f"✓ FactSet client created: {type(factset_client).__name__}")
    
    print("\n5. Testing yfinance fallback...")
    yf_client = DataClientFactory.create(source="yfinance", cache_dir="data/cache")
    print(f"✓ yfinance client created: {type(yf_client).__name__}")
    
    print("\n" + "="*70)
    print("✅ ALL FACTORY TESTS PASSED!")
    print("="*70)
    return True


if __name__ == "__main__":
    print("\nFactSet Client & Factory Test Suite")
    print("="*70)
    
    # Run tests
    factset_success = test_factset_client()
    print("\n")
    factory_success = test_factory()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"FactSet Client: {'✅ PASS' if factset_success else '❌ FAIL'}")
    print(f"Factory Pattern: {'✅ PASS' if factory_success else '❌ FAIL'}")
    print("="*70)
    
    sys.exit(0 if (factset_success or factory_success) else 1)
