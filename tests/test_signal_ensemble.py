"""Unit tests for signal ensemble."""
import pytest
import numpy as np
from strategy.signal_ensemble import SignalEnsemble


class TestSignalEnsembleBasics:
    """Test basic signal computation."""

    def test_initialization(self):
        se = SignalEnsemble(zscore_window=60, momentum_window=5)
        assert se.zscore_window == 60
        assert se.momentum_window == 5

    def test_compute_signals_short_history(self):
        """Signals should handle short history gracefully."""
        se = SignalEnsemble(zscore_window=60)
        short_spread = np.array([1.0, 2.0, 3.0])  # less than zscore_window
        signals = se.compute_signals(short_spread)
        assert len(signals) == 4  # all signal types present
        for key in ["zscore", "momentum", "zscore_momentum", "spread_level"]:
            assert key in signals
            assert isinstance(signals[key], float)

    def test_compute_signals_returns_dict(self):
        """Compute signals should return properly keyed dict."""
        se = SignalEnsemble()
        spread = np.random.randn(100)
        signals = se.compute_signals(spread)
        expected_keys = {"zscore", "momentum", "zscore_momentum", "spread_level"}
        assert set(signals.keys()) == expected_keys

    def test_zero_spread_gives_zero_zscore(self):
        """Constant spread should give zero zscore."""
        se = SignalEnsemble()
        spread = np.ones(100)  # constant
        signals = se.compute_signals(spread)
        assert abs(signals["zscore"]) < 0.01

    def test_positive_excursion_gives_positive_zscore(self):
        """Positive spread excursion should give positive zscore."""
        se = SignalEnsemble(zscore_window=20)
        # Create deterministic: base variation with variance, then step up
        spread = np.concatenate([
            np.linspace(-1.0, 1.0, 50),  # base variation
            np.linspace(1.5, 2.5, 20)  # rising excursion at end
        ])
        signals = se.compute_signals(spread)
        assert signals["zscore"] > 0.3  # should be positive in last window

    def test_negative_excursion_gives_negative_zscore(self):
        """Negative spread excursion should give negative zscore."""
        se = SignalEnsemble(zscore_window=20)
        #Create deterministic: base variation, then drop down
        spread = np.concatenate([
            np.linspace(-1.0, 1.0, 50),  # base variation
            np.linspace(-1.5, -2.5, 20)  # falling excursion at end
        ])
        signals = se.compute_signals(spread)
        assert signals["zscore"] < -0.3  # should be negative in last window


class TestSignalStrength:
    """Test signal combination."""

    def test_signal_strength_combines_signals(self):
        """Combined signal should reflect zscore mainly."""
        se = SignalEnsemble()
        signals = {
            "zscore": 3.0,
            "momentum": 0.0,
            "zscore_momentum": 0.0,
            "spread_level": 0.5,
        }
        strength = se.signal_strength(signals)
        # z-score=3 normalized by 3 = 1.0, weight 0.6 → contribution 0.6
        assert strength > 0.5
        assert strength <= 1.0

    def test_signal_strength_normalizes_negative(self):
        """Negative signals should produce negative strength."""
        se = SignalEnsemble()
        signals = {
            "zscore": -3.0,
            "momentum": 0.0,
            "zscore_momentum": 0.0,
            "spread_level": 0.5,
        }
        strength = se.signal_strength(signals)
        assert strength < -0.5
        assert strength >= -1.0

    def test_signal_strength_clips_to_range(self):
        """Signal strength should stay in [-1, 1]."""
        se = SignalEnsemble()
        signals = {
            "zscore": 100.0,  # extreme
            "momentum": 100.0,
            "zscore_momentum": 100.0,
            "spread_level": 0.5,
        }
        strength = se.signal_strength(signals)
        assert -1.0 <= strength <= 1.0

    def test_custom_weights(self):
        """Custom weights should change signal strength."""
        se = SignalEnsemble()
        signals = {
            "zscore": 3.0,
            "momentum": -3.0,  # opposite signal
            "zscore_momentum": 0.0,
            "spread_level": 0.5,
        }
        # Equal weights → cancellation
        strength_equal = se.signal_strength(
            signals,
            weights={"zscore": 0.5, "momentum": 0.5, "zscore_momentum": 0.0},
        )
        # Zscore dominant
        strength_zscore = se.signal_strength(
            signals, weights={"zscore": 1.0, "momentum": 0.0, "zscore_momentum": 0.0}
        )
        assert abs(strength_zscore) > abs(strength_equal)


class TestEntrySignal:
    """Test entry generation."""

    def test_entry_long_on_negative_zscore(self):
        """Large negative zscore should trigger long entry."""
        se = SignalEnsemble()
        signals = {"zscore": -2.5, "momentum": 0.0, "zscore_momentum": 0.0}
        direction = se.entry_signal(signals, entry_threshold=2.0, momentum_confirm=False)
        assert direction == "long_spread"

    def test_entry_short_on_positive_zscore(self):
        """Large positive zscore should trigger short entry."""
        se = SignalEnsemble()
        signals = {"zscore": 2.5, "momentum": 0.0, "zscore_momentum": 0.0}
        direction = se.entry_signal(signals, entry_threshold=2.0, momentum_confirm=False)
        assert direction == "short_spread"

    def test_no_entry_below_threshold(self):
        """Zscore below threshold should not trigger entry."""
        se = SignalEnsemble()
        signals = {"zscore": 0.5, "momentum": 0.0, "zscore_momentum": 0.0}
        direction = se.entry_signal(signals, entry_threshold=2.0)
        assert direction is None

    def test_momentum_confirmation_blocks_entry(self):
        """With momentum_confirm=True, wrong momentum should block entry."""
        se = SignalEnsemble()
        signals = {"zscore": -2.5, "momentum": 1.0}  # negative zscore but positive momentum
        # Without confirmation
        direction = se.entry_signal(signals, momentum_confirm=False)
        assert direction == "long_spread"
        # With confirmation
        direction = se.entry_signal(signals, momentum_confirm=True)
        assert direction is None

    def test_momentum_confirmation_allows_entry(self):
        """With momentum_confirm=True and matching momentum, entry allowed."""
        se = SignalEnsemble()
        signals = {"zscore": -2.5, "momentum": -0.1}  # both negative (mean-revert up)
        direction = se.entry_signal(signals, momentum_confirm=True)
        assert direction == "long_spread"


class TestExitSignal:
    """Test exit generation."""

    def test_exit_on_small_zscore(self):
        """Small zscore should trigger profit-taking exit."""
        se = SignalEnsemble()
        signals = {"zscore": 0.3}  # small
        direction = se.exit_signal(signals, exit_threshold=0.5)
        assert direction == "flat"

    def test_exit_on_large_zscore(self):
        """Large zscore should trigger stop-loss exit."""
        se = SignalEnsemble()
        signals = {"zscore": 4.0}  # large
        direction = se.exit_signal(signals, stop_loss_threshold=3.5)
        assert direction == "flat"

    def test_no_exit_in_middle_band(self):
        """Zscore in [exit, stop_loss] range should not trigger exit."""
        se = SignalEnsemble()
        signals = {"zscore": 2.0}
        direction = se.exit_signal(signals, exit_threshold=0.5, stop_loss_threshold=3.5)
        assert direction is None


class TestMomentumComputation:
    """Test momentum signal."""

    def test_positive_momentum_on_rising_spread(self):
        """Rising spread should give positive momentum."""
        se = SignalEnsemble(momentum_window=5)
        # Long enough history for zscore and momentum computation
        spread = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        signals = se.compute_signals(spread)
        assert signals["momentum"] > 0.8  # mean difference should be ~1.0

    def test_negative_momentum_on_falling_spread(self):
        """Falling spread should give negative momentum."""
        se = SignalEnsemble(momentum_window=5)
        # Long enough history for zscore and momentum computation
        spread = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        signals = se.compute_signals(spread)
        assert signals["momentum"] < -0.8  # mean difference should be ~-1.0
