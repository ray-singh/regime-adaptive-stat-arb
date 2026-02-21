"""
FactSet Data Client for Regime-Adaptive Statistical Arbitrage Platform

Provides a robust interface for fetching historical OHLCV data from FactSet
with error handling, retry logic, and efficient bulk downloads.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactSetClient:
    """
    Client for fetching and managing historical market data from FactSet.
    
    Features:
    - Bulk ticker downloads with progress tracking
    - Automatic retry logic for failed requests
    - Data validation and cleaning
    - Optional local caching to Parquet format
    - Support for adjusted prices
    
    Note: Requires FACTSET_API_KEY environment variable.
    """
    
    BASE_URL = "https://api.factset.com/content/factset-prices/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.2
    ):
        """
        Initialize the FactSet client.
        
        Args:
            api_key: FactSet API key (defaults to FACTSET_API_KEY env var)
            cache_dir: Directory for caching data as Parquet files
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Base delay between retries (exponential backoff)
            rate_limit_delay: Delay between successive ticker requests
        """
        self.api_key = api_key or os.getenv("FACTSET_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FactSet API key not found. Set FACTSET_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        
        # Setup session with authentication
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
    def fetch_ticker(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "10y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime)
            period: Period to download (e.g., '1d', '5d', '1mo', '10y')
            interval: Data interval ('1d' only supported currently)
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
        
        # Convert period to date range if needed
        if not start_date or not end_date:
            end_date = datetime.now()
            start_date = self._period_to_start_date(period, end_date)
        
        # Format dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Fetch from FactSet with retry logic
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Fetching {ticker} from FactSet (attempt {attempt + 1}/{self.retry_attempts})")
                
                # FactSet Prices API endpoint
                url = f"{self.BASE_URL}/prices"
                
                params = {
                    "ids": ticker,
                    "startDate": start_str,
                    "endDate": end_str,
                    "frequency": "D",  # Daily
                    "calendar": "FIVEDAY",  # Trading days only
                    "adjust": "SPLIT_ADJ"  # Split-adjusted prices
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or "data" not in data:
                    logger.warning(f"No data returned for {ticker}")
                    return pd.DataFrame()
                
                # Parse response into DataFrame
                df = self._parse_factset_response(data, ticker)
                
                if df.empty:
                    logger.warning(f"Empty DataFrame for {ticker}")
                    return pd.DataFrame()
                
                # Clean and standardize
                df = self._clean_data(df, ticker)
                
                # Cache if enabled
                if self.cache_dir:
                    self._save_to_cache(df, ticker, interval)
                
                time.sleep(self.rate_limit_delay)
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    logger.error(f"Authentication failed. Check FACTSET_API_KEY.")
                    raise
                elif e.response.status_code == 429:
                    logger.warning(f"Rate limit exceeded for {ticker}")
                    if attempt < self.retry_attempts - 1:
                        delay = self.retry_delay * (2 ** attempt) * 2  # Longer delay for rate limits
                        logger.info(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed to fetch {ticker} after {self.retry_attempts} attempts")
                        return pd.DataFrame()
                else:
                    logger.warning(f"HTTP error fetching {ticker}: {str(e)}")
                    if attempt < self.retry_attempts - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        return pd.DataFrame()
                        
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
        period: str = "10y"
    ) -> pd.DataFrame:
        """
        Fetch multiple tickers using FactSet's batch API (if available).
        Falls back to sequential fetching if batch not supported.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            period: Period to download
            
        Returns:
            DataFrame with multi-ticker OHLCV data
        """
        # FactSet batch API can handle multiple IDs in a single request
        try:
            # Convert period to date range if needed
            if not start_date or not end_date:
                end_date = datetime.now()
                start_date = self._period_to_start_date(period, end_date)
            
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            logger.info(f"Fetching {len(tickers)} tickers using FactSet batch API...")
            
            # FactSet can handle multiple IDs separated by comma
            # But has limits, so we'll batch in chunks of 50
            chunk_size = 50
            all_data = []
            
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]
                ids_str = ",".join(chunk)
                
                url = f"{self.BASE_URL}/prices"
                params = {
                    "ids": ids_str,
                    "startDate": start_str,
                    "endDate": end_str,
                    "frequency": "D",
                    "calendar": "FIVEDAY",
                    "adjust": "SPLIT_ADJ"
                }
                
                response = self.session.get(url, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                if data and "data" in data:
                    df_chunk = self._parse_factset_batch_response(data)
                    if not df_chunk.empty:
                        all_data.append(df_chunk)
                
                logger.info(f"Fetched batch {i//chunk_size + 1}/{(len(tickers)-1)//chunk_size + 1}")
                time.sleep(self.rate_limit_delay)
            
            if not all_data:
                logger.error("No data fetched in batch mode")
                return pd.DataFrame()
            
            combined = pd.concat(all_data, axis=0, ignore_index=True)
            combined = combined.sort_values(['ticker', 'Date']).reset_index(drop=True)
            
            logger.info(f"Batch download complete: {len(combined):,} rows")
            return combined
            
        except Exception as e:
            logger.error(f"Batch download failed: {str(e)}")
            logger.info("Falling back to sequential fetch...")
            return self.fetch_bulk(tickers, start_date, end_date, period=period)
    
    def _parse_factset_response(self, response_data: Dict, ticker: str) -> pd.DataFrame:
        """Parse single ticker FactSet API response."""
        try:
            records = response_data.get("data", [])
            if not records:
                return pd.DataFrame()
            
            rows = []
            for record in records:
                row = {
                    "Date": record.get("date"),
                    "open": record.get("priceOpen"),
                    "high": record.get("priceHigh"),
                    "low": record.get("priceLow"),
                    "close": record.get("price"),  # Unadjusted close
                    "adj_close": record.get("priceAdj"),  # Adjusted close
                    "volume": record.get("volume"),
                    "ticker": ticker
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            return df
            
        except Exception as e:
            logger.error(f"Error parsing FactSet response for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _parse_factset_batch_response(self, response_data: Dict) -> pd.DataFrame:
        """Parse batch FactSet API response with multiple tickers."""
        try:
            records = response_data.get("data", [])
            if not records:
                return pd.DataFrame()
            
            rows = []
            for record in records:
                ticker = record.get("fsymId", "").split("-")[0]  # Extract ticker from fsymId
                row = {
                    "Date": record.get("date"),
                    "open": record.get("priceOpen"),
                    "high": record.get("priceHigh"),
                    "low": record.get("priceLow"),
                    "close": record.get("price"),
                    "adj_close": record.get("priceAdj"),
                    "volume": record.get("volume"),
                    "ticker": ticker
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            return df
            
        except Exception as e:
            logger.error(f"Error parsing FactSet batch response: {str(e)}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and standardize fetched data."""
        df = df.copy()
        
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Remove rows with null prices
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure ticker column exists
        if 'ticker' not in df.columns:
            df['ticker'] = ticker
        
        # Reorder columns
        cols = ['Date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[[col for col in cols if col in df.columns]]
        
        return df
    
    def _period_to_start_date(self, period: str, end_date: datetime) -> datetime:
        """Convert period string to start date."""
        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
            "10y": timedelta(days=3650),
            "max": timedelta(days=10950)  # ~30 years
        }
        
        delta = period_map.get(period, timedelta(days=3650))
        return end_date - delta
    
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
            cache_file = self.cache_dir / f"{ticker}_{interval}_factset.parquet"
            df.to_parquet(cache_file, index=False, compression='snappy')
            logger.debug(f"Cached {ticker} to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {str(e)}")
    
    def _load_from_cache(self, ticker: str, interval: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet cache."""
        if self.cache_dir is None:
            return None
        
        try:
            cache_file = self.cache_dir / f"{ticker}_{interval}_factset.parquet"
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
