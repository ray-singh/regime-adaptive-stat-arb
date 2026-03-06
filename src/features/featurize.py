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

        # Moving average deviation: % price deviation from rolling mean (captures trend direction)
        rolling_ma = df[price_col].rolling(window=w, min_periods=max(1, w//2)).mean()
        df[f'ma_dev_{w}'] = (df[price_col] - rolling_ma) / rolling_ma.replace(0, np.nan)

        # Trend strength: |momentum| / realized_vol — high = trending, low = mean-reverting
        df[f'trend_str_{w}'] = df[f'mom_{w}'].abs() / df[f'rv_{w}'].replace(0, np.nan)

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


def compute_market_features(wide_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Compute cross-asset market-wide features from a wide (Date × Ticker) price matrix.

    Produces (spec §3.2 — market-wide features):
        - ``ret_dispersion``  : cross-sectional std of daily returns (measures divergence / crisis)
        - ``avg_rv``          : average realized volatility across all assets
        - ``avg_corr``        : average correlation of each asset with the equal-weight market return
        - ``mom_dispersion``  : cross-sectional std of rolling momentum (sector divergence)

    Parameters
    ----------
    wide_df : DataFrame
        Wide price matrix (index=DatetimeIndex, columns=tickers).
    window : int
        Rolling window for vol and correlation estimates.

    Returns
    -------
    DataFrame with DatetimeIndex and the four feature columns above.
    """
    rets = wide_df.pct_change()
    features = pd.DataFrame(index=wide_df.index)

    # Cross-sectional return dispersion: how much stocks diverge each day
    features['ret_dispersion'] = rets.std(axis=1)

    # Average realized volatility across all assets (annualized)
    rolling_rv = rets.rolling(window, min_periods=max(1, window // 2)).std() * np.sqrt(252)
    features['avg_rv'] = rolling_rv.mean(axis=1)

    # Average correlation with equal-weight market return (proxy for systematic risk)
    market_ret = rets.mean(axis=1)
    corr_cols = {}
    for col in rets.columns:
        corr_cols[col] = rets[col].rolling(window, min_periods=max(1, window // 2)).corr(market_ret)
    features['avg_corr'] = pd.DataFrame(corr_cols).mean(axis=1)

    # Cross-sectional dispersion of rolling momentum (high = uneven sector performance = trending)
    rolling_mom = wide_df.pct_change(window)
    features['mom_dispersion'] = rolling_mom.std(axis=1)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return features
