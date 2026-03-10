"""Unit tests for VolatilityRegimeDetector."""
import numpy as np
import pandas as pd
import pytest

from regime.volatility_detector import VolatilityRegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataFrame with a 'rv_20' column and a 'logret' column."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    rv = np.abs(rng.normal(0.015, 0.008, n)) + 0.001   # always positive
    logret = rng.normal(0.0002, rv)
    return pd.DataFrame({"rv_20": rv, "logret": logret}, index=dates)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_is_3_states(self):
        d = VolatilityRegimeDetector()
        assert d.n_regimes == 3

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_valid_n_states(self, n):
        d = VolatilityRegimeDetector(n_states=n)
        assert d.n_regimes == n

    @pytest.mark.parametrize("n", [1, 0, 5, -1])
    def test_invalid_n_states_raises(self, n):
        with pytest.raises(ValueError):
            VolatilityRegimeDetector(n_states=n)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_sets_thresholds(self):
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df()
        d.fit(df)
        assert d._thresholds is not None
        # n_states=3 → 2 interior quantile thresholds
        assert len(d._thresholds) == 2

    def test_thresholds_are_sorted_ascending(self):
        d = VolatilityRegimeDetector(n_states=4)
        df = _make_df()
        d.fit(df)
        assert list(d._thresholds) == sorted(d._thresholds)

    def test_fit_missing_column_raises(self):
        d = VolatilityRegimeDetector(rv_col="rv_20")
        df = pd.DataFrame({"other_col": [0.1, 0.2, 0.3]})
        with pytest.raises(KeyError):
            d.fit(df)

    def test_fit_returns_self(self):
        d = VolatilityRegimeDetector()
        df = _make_df()
        result = d.fit(df)
        assert result is d


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_before_fit_raises(self):
        d = VolatilityRegimeDetector()
        df = _make_df()
        with pytest.raises(RuntimeError):
            d.predict(df)

    def test_predict_returns_series(self):
        d = VolatilityRegimeDetector()
        df = _make_df()
        d.fit(df)
        labels = d.predict(df)
        assert isinstance(labels, pd.Series)
        assert len(labels) == len(df)

    def test_labels_in_valid_range(self):
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df()
        d.fit(df)
        labels = d.predict(df)
        assert labels.min() >= 0
        assert labels.max() <= d.n_states - 1

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_all_regimes_represented(self, n):
        """With enough data, all n regimes should appear at least once."""
        d = VolatilityRegimeDetector(n_states=n)
        df = _make_df(n=1000)
        d.fit(df)
        labels = d.predict(df)
        assert labels.nunique() == n

    def test_high_rv_gets_high_regime_label(self):
        """Rows with very high rv should all fall in the highest regime bin."""
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df(n=500)
        d.fit(df)

        # Rows with rv far above all thresholds
        extreme = pd.DataFrame(
            {"rv_20": [1.0, 2.0, 3.0]},   # >> any plausible threshold
            index=pd.date_range("2030-01-01", periods=3, freq="B"),
        )
        labels = d.predict(extreme)
        assert (labels == d.n_states - 1).all()

    def test_low_rv_gets_low_regime_label(self):
        """Rows with rv near zero should all fall in regime 0."""
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df(n=500)
        d.fit(df)

        low = pd.DataFrame(
            {"rv_20": [0.0001, 0.0001, 0.0001]},
            index=pd.date_range("2030-01-01", periods=3, freq="B"),
        )
        labels = d.predict(low)
        assert (labels == 0).all()


# ---------------------------------------------------------------------------
# state_stats()
# ---------------------------------------------------------------------------

class TestStateStats:
    def test_state_stats_returns_one_row_per_regime(self):
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df(n=500)
        d.fit(df)
        labels = d.predict(df)
        stats = d.state_stats(df, labels)
        assert len(stats) == 3

    def test_state_stats_includes_rv_mean(self):
        d = VolatilityRegimeDetector(n_states=2)
        df = _make_df(n=500)
        d.fit(df)
        labels = d.predict(df)
        stats = d.state_stats(df, labels)
        assert "rv_mean" in stats.columns

    def test_state_stats_includes_ann_ret_when_logret_present(self):
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df(n=500)
        d.fit(df)
        labels = d.predict(df)
        stats = d.state_stats(df, labels)
        assert "ann_ret_pct" in stats.columns

    def test_rv_mean_increases_by_regime(self):
        """Higher regime labels should have higher mean realised volatility."""
        d = VolatilityRegimeDetector(n_states=3)
        df = _make_df(n=1000)
        d.fit(df)
        labels = d.predict(df)
        stats = d.state_stats(df, labels)
        rv_means = stats["rv_mean"].values
        assert rv_means[0] < rv_means[1] < rv_means[2]
