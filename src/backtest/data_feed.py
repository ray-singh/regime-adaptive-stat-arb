"""Historical bar data feed.

Yields one MarketEvent per trading day, advancing through the price matrix in
calendar order.  Provides a rolling window of past prices for feature/z-score
computation.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator, Optional
import pandas as pd

from .events import MarketEvent


class HistoricalDataFeed:
    """Wraps a wide (Date × Ticker) price DataFrame into a bar-by-bar generator.

    Parameters
    ----------
    price_df : DataFrame
        Wide price matrix (index=DatetimeIndex, columns=tickers).
    ohlcv_panels : dict[str, DataFrame], optional
        Per-ticker long-format OHLCV DataFrames, keyed by ticker.
        Used to supply full bar `ohlcv` attribute on each MarketEvent.
    start : str or Timestamp, optional
        First date to emit (inclusive).
    end : str or Timestamp, optional
        Last date to emit (inclusive).
    warmup_bars : int
        Number of bars to consume *silently* for indicator warm-up before
        emitting the first MarketEvent.  These bars are still accessible via
        `history`.
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        ohlcv_panels: Optional[dict[str, pd.DataFrame]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        warmup_bars: int = 0,
    ):
        df = price_df.copy()
        # Normalize index to timezone-naive DatetimeIndex safely.
        idx = pd.to_datetime(df.index)
        try:
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            else:
                idx = idx.tz_localize(None)
        except Exception:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                try:
                    idx = idx.tz_convert(None)
                except Exception:
                    pass
        df.index = idx
        df = df.sort_index()

        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]

        self._df = df
        self._tickers = list(df.columns)
        self._dates = list(df.index)
        self._ohlcv_panels = ohlcv_panels or {}
        self._warmup_bars = warmup_bars

        # Rolling history: deque of (date, price_dict) — most recent last
        self._history: deque[tuple[pd.Timestamp, dict[str, float]]] = deque()
        self._current_idx: int = -1

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return self._tickers

    @property
    def dates(self) -> list[pd.Timestamp]:
        return self._dates[self._warmup_bars:]

    @property
    def current_date(self) -> Optional[pd.Timestamp]:
        if self._current_idx < 0:
            return None
        return self._dates[self._current_idx]

    def current_prices(self) -> dict[str, float]:
        """Return the most recent close prices as a dict."""
        if not self._history:
            return {}
        return dict(self._history[-1][1])

    def price_history(self, ticker: str, n: int) -> pd.Series:
        """Return the last n close prices for a ticker as a Series."""
        dates, prices = zip(*[(d, p.get(ticker, float("nan"))) for d, p in self._history])
        s = pd.Series(prices, index=dates, name=ticker)
        return s.iloc[-n:] if n < len(s) else s

    def history_df(self, n: Optional[int] = None) -> pd.DataFrame:
        """Return history as a wide price DataFrame (most recent last)."""
        dates  = [d for d, _ in self._history]
        prices = [p for _, p in self._history]
        df = pd.DataFrame(prices, index=dates)
        return df if n is None else df.iloc[-n:]

    # -----------------------------------------------------------------------
    # Generator
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[MarketEvent]:
        for idx, date in enumerate(self._dates):
            row = self._df.iloc[idx]
            prices = {t: float(row[t]) for t in self._tickers if pd.notna(row[t])}
            self._current_idx = idx
            self._history.append((date, prices))

            # Still in warmup window — don't emit an event
            if idx < self._warmup_bars:
                continue

            # Build optional full-bar OHLCV snapshot
            ohlcv_snap = None
            if self._ohlcv_panels:
                ohlcv_snap = {}
                for t, panel in self._ohlcv_panels.items():
                    if date in panel.index:
                        ohlcv_snap[t] = panel.loc[date].to_dict()

            yield MarketEvent(date=date, prices=prices, ohlcv=ohlcv_snap)

    def __len__(self) -> int:
        return max(0, len(self._dates) - self._warmup_bars)
