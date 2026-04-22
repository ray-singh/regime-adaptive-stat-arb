import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DataClientFactory:
    @staticmethod
    def create(
        cache_dir: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        rate_limit_delay: Optional[float] = None,
        **kwargs
    ):
        from yfinance_client import YFinanceClient
        return YFinanceClient(
            cache_dir=cache_dir,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
            rate_limit_delay=rate_limit_delay or 0.1,
            **kwargs
        )
