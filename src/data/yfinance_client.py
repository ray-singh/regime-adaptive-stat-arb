"""
YFinance Data Client for Regime-Adaptive Statistical Arbitrage Platform

Provides a robust interface for fetching historical OHLCV data from Yahoo Finance
with error handling, retry logic, and efficient bulk downloads.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceClient:
    """
    Client for fetching and managing historical market data from Yahoo Finance.
    
    Features:
    - Bulk ticker downloads with progress tracking
    - Automatic retry logic for failed requests
    - Data validation and cleaning
    - Optional local caching to Parquet format
    - Support for adjustments (splits, dividends)
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.1
    ):
        """
        Initialize the YFinance client.
        
        Args:
            cache_dir: Directory for caching data as Parquet files
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            rate_limit_delay: Delay between successive ticker requests
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        
    def fetch_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "10y",
        interval: str = "1d",
        auto_adjust: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            period: Period to download (e.g., '1d', '5d', '1mo', '10y')
            interval: Data interval ('1d', '1h', '1m', etc.)
            auto_adjust: Adjust OHLC for splits and dividends
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with OHLCV data and ticker column
        """
        # Check cache first
        if use_cache and self.cache_dir:
            cached_data = self._load_from_cache(ticker, interval)
            if cached_data is not None:
                logger.info(f"Loaded {ticker} from cache")
                return self._filter_by_date(cached_data, start_date, end_date)
        
        # Fetch from yfinance with retry logic
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Fetching {ticker} (attempt {attempt + 1}/{self.retry_attempts})")
                
                ticker_obj = yf.Ticker(ticker)
                
                if start_date and end_date:
                    df = ticker_obj.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=auto_adjust
                    )
                else:
                    df = ticker_obj.history(
                        period=period,
                        interval=interval,
                        auto_adjust=auto_adjust
                    )
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()
                
                # Add ticker column and clean data
                df = self._clean_data(df, ticker)
                
                # Cache if enabled
                if self.cache_dir:
                    self._save_to_cache(df, ticker, interval)
                
                time.sleep(self.rate_limit_delay)
                return df
                
            except Exception as e:
                logger.warning(f"Error fetching {ticker}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {ticker} after {self.retry_attempts} attempts")
                    return pd.DataFrame()
    
    def fetch_bulk(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "10y",
        interval: str = "1d",
        auto_adjust: bool = True,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple tickers efficiently.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            period: Period to download
            interval: Data interval
            auto_adjust: Adjust OHLC for splits and dividends
            use_cache: Use cached data if available
            show_progress: Show progress during bulk download
            
        Returns:
            DataFrame with multi-ticker OHLCV data
        """
        all_data = []
        failed_tickers = []
        
        total = len(tickers)
        for idx, ticker in enumerate(tickers, 1):
            if show_progress and idx % 10 == 0:
                logger.info(f"Progress: {idx}/{total} tickers ({idx/total*100:.1f}%)")
            
            df = self.fetch_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                use_cache=use_cache
            )
            
            if not df.empty:
                all_data.append(df)
            else:
                failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to fetch {len(failed_tickers)} tickers: {failed_tickers[:10]}")
        
        if not all_data:
            logger.error("No data fetched for any ticker")
            return pd.DataFrame()
        
        # Concatenate all data
        combined = pd.concat(all_data, axis=0)
        combined = combined.sort_values(['ticker', 'Date']).reset_index(drop=True)
        
        logger.info(f"Successfully fetched {len(all_data)}/{total} tickers")
        logger.info(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
        logger.info(f"Total rows: {len(combined):,}")
        
        return combined
    
    def fetch_bulk_optimized(
        self,
        tickers: List[str],
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "10y",
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers using yfinance's optimized bulk download.
        This is faster than individual fetches but has less control.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            period: Period to download
            auto_adjust: Adjust OHLC for splits and dividends
            
        Returns:
            DataFrame with multi-ticker OHLCV data
        """
        try:
            logger.info(f"Fetching {len(tickers)} tickers using bulk download...")
            
            if start_date and end_date:
                data = yf.download(
                    tickers=tickers,
                    start=start_date,
                    end=end_date,
                    auto_adjust=auto_adjust,
                    group_by='ticker',
                    threads=True,
                    progress=True
                )
            else:
                data = yf.download(
                    tickers=tickers,
                    period=period,
                    auto_adjust=auto_adjust,
                    group_by='ticker',
                    threads=True,
                    progress=True
                )
            
            if data.empty:
                logger.error("No data returned from bulk download")
                return pd.DataFrame()
            
            # Restructure multi-level columns to long format
            dfs = []
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        ticker_data = data.copy()
                    else:
                        ticker_data = data[ticker].copy()
                    
                    if ticker_data.empty:
                        continue
                    
                    ticker_data = ticker_data.reset_index()
                    ticker_data['ticker'] = ticker
                    ticker_data.columns = [col.lower().replace(' ', '_') for col in ticker_data.columns]
                    
                    # Rename Date column if needed
                    if 'date' in ticker_data.columns:
                        ticker_data.rename(columns={'date': 'Date'}, inplace=True)
                    
                    dfs.append(ticker_data)
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {str(e)}")
                    continue
            
            if not dfs:
                logger.error("No valid data after processing")
                return pd.DataFrame()
            
            combined = pd.concat(dfs, axis=0, ignore_index=True)
            combined = self._standardize_columns(combined)
            
            logger.info(f"Bulk download complete: {len(combined):,} rows")
            return combined
            
        except Exception as e:
            logger.error(f"Bulk download failed: {str(e)}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and standardize fetched data."""
        df = df.copy()
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Standardize column names
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df.rename(columns={'date': 'Date'}, inplace=True)
        
        # Add ticker column
        df['ticker'] = ticker
        
        # Remove rows with null prices
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure proper datatypes
        df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent column naming and ordering."""
        expected_cols = ['Date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        
        # Keep only expected columns that exist
        existing_cols = [col for col in expected_cols if col in df.columns]
        df = df[existing_cols]
        
        return df
    
    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[Union[str, datetime]],
        end_date: Optional[Union[str, datetime]]
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if df.empty:
            return df
        
        df = df.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['Date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['Date'] <= end_date]
        
        return df
    
    def _save_to_cache(self, df: pd.DataFrame, ticker: str, interval: str):
        """Save DataFrame to Parquet cache."""
        if self.cache_dir is None or df.empty:
            return
        
        try:
            cache_file = self.cache_dir / f"{ticker}_{interval}.parquet"
            df.to_parquet(cache_file, index=False, compression='snappy')
            logger.debug(f"Cached {ticker} to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {str(e)}")
    
    def _load_from_cache(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet cache."""
        if self.cache_dir is None:
            return None
        
        try:
            cache_file = self.cache_dir / f"{ticker}_{interval}.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {str(e)}")
        
        return None
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for fetched data."""
        if df.empty:
            return {}
        
        summary = {
            'total_rows': len(df),
            'unique_tickers': df['ticker'].nunique(),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'trading_days': df.groupby('ticker')['Date'].count().describe().to_dict(),
            'missing_data': df.isnull().sum().to_dict()
        }
        
        return summary