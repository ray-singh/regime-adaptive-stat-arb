"""Fetch full 200-stock universe from FactSet (resumable).

Writes per-ticker Parquet files to `data/cache_factset/` and optionally a combined file.

This script requires FACTSET_API_KEY to be set in the environment.
"""

import os
from pathlib import Path
import sys
import time
from dotenv import load_dotenv
from data_client_factory import DataClientFactory
from universe import get_universe


def main(
    cache_dir: str = "data/cache_factset",
    period: str = "10y",
    interval: str = "1d",
    rate_limit_delay: float = 0.2,
    combine_output: bool = True,
):
    load_dotenv()
    
    # Check API key
    api_key = os.getenv("FACTSET_API_KEY")
    if not api_key:
        print("ERROR: FACTSET_API_KEY not found in environment. Set it and re-run.")
        sys.exit(2)

    out_dir = Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = DataClientFactory.create(source="factset", cache_dir=str(out_dir), rate_limit_delay=rate_limit_delay)

    universe = get_universe("top200")
    print(f"Starting fetch for {len(universe)} tickers into {out_dir}")

    fetched = []
    skipped = []

    for i, ticker in enumerate(universe, 1):
        out_file = out_dir / f"{ticker}_factset.parquet"
        if out_file.exists():
            skipped.append(ticker)
            print(f"[{i}/{len(universe)}] Skipping {ticker} (cached)")
            continue

        print(f"[{i}/{len(universe)}] Fetching {ticker}...")
        try:
            df = client.fetch_ticker(ticker, period=period, interval=interval)
            if df.empty:
                print(f"  -> No data for {ticker}, skipping")
                continue

            df.to_parquet(out_file, index=False, compression="snappy")
            fetched.append(ticker)
            print(f"  -> Saved {out_file} ({len(df)} rows)")
        except Exception as e:
            print(f"  -> Error fetching {ticker}: {e}")
        
        # small delay to respect rate limits
        time.sleep(rate_limit_delay)

    print(f"\nFetch complete. fetched={len(fetched)}, skipped={len(skipped)}")

    if combine_output and fetched:
        try:
            import pandas as pd
            print("Combining per-ticker files into data/full_universe_factset_10y.parquet (may take time)")
            parts = []
            for t in universe:
                p = out_dir / f"{t}_factset.parquet"
                if p.exists():
                    parts.append(pd.read_parquet(p))
            if parts:
                combined = pd.concat(parts, axis=0, ignore_index=True)
                combined.to_parquet("data/full_universe_factset_10y.parquet", index=False, compression="snappy")
                print(f"Combined file written: data/full_universe_factset_10y.parquet ({len(combined):,} rows)")
        except Exception as e:
            print(f"Failed to combine files: {e}")


if __name__ == "__main__":
    main()
