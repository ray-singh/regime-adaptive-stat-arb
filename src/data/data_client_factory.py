"""
Data Client Factory for Regime-Adaptive Statistical Arbitrage Platform

Provides a unified interface for creating data clients with automatic fallback.
"""

import os
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

DataSource = Literal["factset", "yfinance", "auto"]


class DataClientFactory:
    """
    Factory for creating data clients with automatic fallback capabilities.
    
    Usage:
        # Auto-detect (tries FactSet first, falls back to yfinance)
        client = DataClientFactory.create()
        
        # Explicit provider
        client = DataClientFactory.create(source="factset")
        
        # With custom config
        client = DataClientFactory.create(
            source="auto",
            cache_dir="data/cache",
            retry_attempts=5
        )
    """
    
    @staticmethod
    def create(
        source: DataSource = "auto",
        cache_dir: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: Optional[float] = None,
        **kwargs
    ):
        """
        Create a data client with automatic fallback.
        
        Args:
            source: Data source ("factset", "yfinance", or "auto")
            cache_dir: Directory for caching data
            retry_attempts: Number of retry attempts
            retry_delay: Base delay between retries
            rate_limit_delay: Delay between requests (auto-set per provider)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Data client instance (FactSetClient or YFinanceClient)
        """
        if source == "auto":
            # Try FactSet first if API key is available
            if os.getenv("FACTSET_API_KEY"):
                logger.info("FACTSET_API_KEY found, using FactSet as primary data source")
                try:
                    return DataClientFactory._create_factset(
                        cache_dir, retry_attempts, retry_delay, rate_limit_delay, **kwargs
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize FactSet client: {str(e)}")
                    logger.info("Falling back to yfinance")
            
            # Fallback to yfinance
            logger.info("Using yfinance as data source")
            return DataClientFactory._create_yfinance(
                cache_dir, retry_attempts, retry_delay, rate_limit_delay, **kwargs
            )
        
        elif source == "factset":
            return DataClientFactory._create_factset(
                cache_dir, retry_attempts, retry_delay, rate_limit_delay, **kwargs
            )
        
        elif source == "yfinance":
            return DataClientFactory._create_yfinance(
                cache_dir, retry_attempts, retry_delay, rate_limit_delay, **kwargs
            )
        
        else:
            raise ValueError(f"Unknown data source: {source}. Use 'factset', 'yfinance', or 'auto'")
    
    @staticmethod
    def _create_factset(
        cache_dir: Optional[str],
        retry_attempts: int,
        retry_delay: float,
        rate_limit_delay: Optional[float],
        **kwargs
    ):
        """Create FactSet client."""
        from data.factset_client import FactSetClient
        
        return FactSetClient(
            cache_dir=cache_dir,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            rate_limit_delay=rate_limit_delay or 0.2,  # FactSet default
            **kwargs
        )
    
    @staticmethod
    def _create_yfinance(
        cache_dir: Optional[str],
        retry_attempts: int,
        retry_delay: float,
        rate_limit_delay: Optional[float],
        **kwargs
    ):
        """Create yfinance client."""
        from data.yfinance_client import YFinanceClient
        
        return YFinanceClient(
            cache_dir=cache_dir,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            rate_limit_delay=rate_limit_delay or 0.1,  # yfinance default
            **kwargs
        )
    
    @staticmethod
    def get_available_sources() -> list[str]:
        """Get list of available data sources based on credentials."""
        sources = ["yfinance"]  # Always available
        
        if os.getenv("FACTSET_API_KEY"):
            sources.insert(0, "factset")  # Prefer FactSet if available
        
        return sources
    
    @staticmethod
    def is_source_available(source: str) -> bool:
        """Check if a specific data source is available."""
        if source == "yfinance":
            return True
        elif source == "factset":
            return bool(os.getenv("FACTSET_API_KEY"))
        else:
            return False
