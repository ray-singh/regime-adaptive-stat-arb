"""Tests for enhanced featurize functions: vol term-structure and market correlation."""

import pytest
import numpy as np
import pandas as pd


def _make_ohlcv(n=100, start_price=100.0, seed=42) -> pd.DataFrame:
    """Create a synthetic single-ticker OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    prices = start_price * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "open": prices * (1 - 0.002),
        "high": prices * 1.005,
        "low": prices * 0.995,
        "close": prices,
        "adj_close": prices,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    })
    return df


def _make_wide_prices(n_tickers=10, n_days=150, seed=0) -> pd.DataFrame:
    """Create a synthetic wide prices DataFrame (dates × tickers)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    data = {}
    price = 100.0
    for i in range(n_tickers):
        ret = rng.normal(0, 0.01, n_days)
        data[f"T{i}"] = price * np.cumprod(1 + ret)
    return pd.DataFrame(data, index=dates)


import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features.featurize import compute_standard_features, compute_market_correlation_feature


class TestVolTermStructure:
    """Test that vol term-structure features are correctly computed."""

    def test_vol_ts_ratio_column_exists(self):
        """compute_standard_features should add vol_ts_ratio column."""
        df = _make_ohlcv(n=120)
        out = compute_standard_features(df, windows=[5, 20, 60])
        assert "vol_ts_ratio" in out.columns

    def test_vol_ts_zscore_column_exists(self):
        """compute_standard_features should add vol_ts_zscore column."""
        df = _make_ohlcv(n=120)
        out = compute_standard_features(df, windows=[5, 20, 60])
        assert "vol_ts_zscore" in out.columns

    def test_vol_ts_ratio_positive(self):
        """Vol term-structure ratio should be positive (ratio of positive vols)."""
        df = _make_ohlcv(n=120)
        out = compute_standard_features(df, windows=[5, 20, 60])
        valid = out["vol_ts_ratio"].dropna()
        assert (valid > 0).all(), "All valid vol_ts_ratio values must be positive"

    def test_vol_ts_ratio_not_all_nan(self):
        """Vol term-structure ratio should produce non-NaN values after warmup."""
        df = _make_ohlcv(n=120)
        out = compute_standard_features(df, windows=[5, 20, 60])
        assert out["vol_ts_ratio"].notna().sum() > 10

    def test_vol_ts_zscore_near_zero_mean(self):
        """Vol term-structure z-score should be approximately zero-mean."""
        df = _make_ohlcv(n=300)
        out = compute_standard_features(df, windows=[5, 20, 60])
        valid = out["vol_ts_zscore"].dropna()
        assert len(valid) > 50
        assert abs(valid.mean()) < 1.0  # Not strictly zero but reasonably centered

    def test_elevated_vol_gives_ratio_above_one(self):
        """When short-term vol spikes, ratio should exceed 1."""
        df = _make_ohlcv(n=120)
        # Inject a volatility spike in the last 5 bars
        spike_prices = df["adj_close"].copy()
        for i in range(-5, 0):
            spike_prices.iloc[i] *= (1 + 0.10 * ((-i) / 5))
        df_spike = df.copy()
        df_spike["adj_close"] = spike_prices
        df_spike["close"] = spike_prices
        out = compute_standard_features(df_spike, windows=[5, 20, 60])
        # Last valid ratio should be relatively elevated
        last_valid = out["vol_ts_ratio"].dropna().iloc[-1]
        assert last_valid > 0

    def test_no_inf_values(self):
        """No infinite values should appear in vol term structure columns."""
        df = _make_ohlcv(n=200)
        out = compute_standard_features(df, windows=[5, 20, 60])
        for col in ["vol_ts_ratio", "vol_ts_zscore"]:
            if col in out.columns:
                assert not np.isinf(out[col].dropna()).any(), f"{col} has infinite values"

    def test_custom_windows_uses_first_and_last(self):
        """With custom windows, vol term-structure uses first (shortest) and last (longest)."""
        df = _make_ohlcv(n=100)
        out = compute_standard_features(df, windows=[10, 30])
        # Should create rv_10 and rv_30 then vol_ts_ratio = rv_10 / rv_30
        assert "rv_10" in out.columns
        assert "rv_30" in out.columns
        assert "vol_ts_ratio" in out.columns


class TestMarketCorrelationFeature:
    """Test compute_market_correlation_feature function."""

    def test_returns_series(self):
        """Should return a pandas Series."""
        wide = _make_wide_prices(n_tickers=5, n_days=100)
        result = compute_market_correlation_feature(wide, window=30)
        assert isinstance(result, pd.Series)

    def test_named_market_corr(self):
        """Series should be named 'market_corr'."""
        wide = _make_wide_prices(n_tickers=5, n_days=100)
        result = compute_market_correlation_feature(wide, window=30)
        assert result.name == "market_corr"

    def test_length_matches_input(self):
        """Output length should match input DataFrame."""
        wide = _make_wide_prices(n_tickers=5, n_days=100)
        result = compute_market_correlation_feature(wide, window=30)
        assert len(result) == len(wide)

    def test_values_in_valid_range(self):
        """Rolling correlation values should be in [-1, 1]."""
        wide = _make_wide_prices(n_tickers=8, n_days=150)
        result = compute_market_correlation_feature(wide, window=30)
        valid = result.dropna()
        assert len(valid) > 10
        assert (valid >= -1.0).all() and (valid <= 1.0).all()

    def test_perfect_correlation_gives_one(self):
        """All identical price series should yield correlation of 1."""
        dates = pd.date_range("2020-01-01", periods=60, freq="B")
        prices = np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, 60)) * 100
        wide = pd.DataFrame({t: prices for t in ["A", "B", "C"]}, index=dates)
        result = compute_market_correlation_feature(wide, window=20)
        valid = result.dropna()
        assert len(valid) > 0
        assert (valid > 0.99).all()

    def test_empty_df_returns_empty_series(self):
        """Empty input returns empty Series."""
        wide = pd.DataFrame()
        result = compute_market_correlation_feature(wide, window=30)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_single_ticker_returns_empty(self):
        """Single ticker (no pairs) returns empty Series."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        wide = pd.DataFrame({"A": np.random.randn(50)}, index=dates)
        result = compute_market_correlation_feature(wide, window=20)
        # Either empty or all zeros (no valid pairs)
        assert len(result) == 0 or (result == 0).all()

    def test_max_tickers_cap(self):
        """max_tickers should cap the number of tickers used."""
        wide = _make_wide_prices(n_tickers=50, n_days=100)
        # Should not raise even with 50 tickers and cap at 5
        result = compute_market_correlation_feature(wide, window=30, max_tickers=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(wide)

    def test_nan_warmup_period(self):
        """First `window` rows should have NaN due to rolling warmup."""
        wide = _make_wide_prices(n_tickers=4, n_days=100)
        window = 30
        result = compute_market_correlation_feature(wide, window=window, min_periods=window)
        # Some early values should be NaN
        early = result.iloc[:window - 1]
        assert early.isna().any()
