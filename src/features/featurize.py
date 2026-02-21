"""Feature computation utilities.

Provides a standard feature set for regime detection and downstream models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def _price_col(df: pd.DataFrame) -> str:
    if 'adj_close' in df.columns:
        return 'adj_close'
    if 'adjusted_close' in df.columns:
        return 'adjusted_close'
    return 'close'


def compute_standard_features(df: pd.DataFrame, windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
    """Compute a standard set of features for a single-ticker OHLCV DataFrame.

    Input must contain at least a `Date` column and a price column (prefers `adj_close`).

    Features added (prefix/fmt):
      - `ret`: simple return
      - `logret`: log return
      - `mom_{w}`: w-day momentum (pct change)
      - `rv_{w}`: realized vol (rolling std of log returns, annualized)
      - `vov_{w}`: vol-of-vol (rolling std of realized vol)
      - `z_{col}_{w}`: rolling z-score of a column

    Returns a DataFrame with the new feature columns.
    """
    if df.empty:
        return df

    df = df.copy()
    price_col = _price_col(df)

    df = df.sort_values('Date').reset_index(drop=True)

    # Returns
    df['ret'] = df[price_col].pct_change()
    df['logret'] = np.log(df[price_col]).diff()

    # Rolling features
    for w in windows:
        # Momentum: pct change over window
        df[f'mom_{w}'] = df[price_col].pct_change(periods=w)

        # Realized vol: rolling std of log returns, annualized by sqrt(252)
        df[f'rv_{w}'] = df['logret'].rolling(window=w, min_periods=max(1, w//2)).std() * np.sqrt(252)

        # Vol of vol: std dev of realized vol over window
        df[f'vov_{w}'] = df[f'rv_{w}'].rolling(window=w, min_periods=max(1, w//2)).std()

        # Rolling mean and std of returns (for zscore)
        rolling_mean = df['ret'].rolling(window=w, min_periods=max(1, w//2)).mean()
        rolling_std = df['ret'].rolling(window=w, min_periods=max(1, w//2)).std()
        df[f'z_ret_{w}'] = (df['ret'] - rolling_mean) / (rolling_std.replace(0, np.nan))

    # Clean infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def rolling_pairwise_correlation(df: pd.DataFrame, left: str, right: str, window: int = 20) -> pd.Series:
    """Compute rolling correlation of returns between two tickers.

    `df` is expected to be a long DataFrame with columns `Date`, `ticker`, and price column.
    `left` and `right` are ticker symbols present in `df['ticker']`.

    Returns a pandas Series indexed by Date containing the rolling correlation.
    """
    price_col = 'adj_close' if 'adj_close' in df.columns else 'close'

    wide = df.pivot(index='Date', columns='ticker', values=price_col)
    if left not in wide.columns or right not in wide.columns:
        raise KeyError(f"Tickers {left} or {right} not present in data")

    # Compute daily returns and rolling corr
    ret = wide.pct_change()
    corr = ret[left].rolling(window=window, min_periods=max(1, window//2)).corr(ret[right])
    return corr
