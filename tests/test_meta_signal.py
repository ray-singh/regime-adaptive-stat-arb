"""Tests for meta-signal model and regime-aware configuration."""

import pytest
import numpy as np
from src.strategy.meta_signal import MetaSignalModel, MetaSignalConfig


class TestMetaSignalConfig:
    """Test MetaSignalConfig initialization and defaults."""

    def test_default_initialization(self):
        """Test default config initialization."""
        cfg = MetaSignalConfig()
        assert cfg.use_learned_weights is False
        assert "zscore" in cfg.fixed_weights
        assert cfg.fixed_weights["zscore"] == 0.6
        assert cfg.fixed_weights["momentum"] == 0.2
        assert cfg.fixed_weights["zscore_momentum"] == 0.2

    def test_regime_thresholds_exist(self):
        """Test that all regime thresholds are defined."""
        cfg = MetaSignalConfig()
        for regime in [0, 1, 2, 3]:
            assert regime in cfg.regime_entry_thresholds
            assert regime in cfg.regime_exit_thresholds
            assert regime in cfg.regime_stop_loss
            assert regime in cfg.regime_strength_scale
            assert regime in cfg.regime_position_scale

    def test_custom_weights(self):
        """Test custom weight initialization."""
        custom_weights = {"zscore": 0.7, "momentum": 0.3, "zscore_momentum": 0.0}
        cfg = MetaSignalConfig(fixed_weights=custom_weights)
        assert cfg.fixed_weights["zscore"] == 0.7
        assert cfg.fixed_weights["momentum"] == 0.3

    def test_bull_regime_thresholds_tighter(self):
        """Test that bull regime has tighter (lower) entry thresholds."""
        cfg = MetaSignalConfig()
        bull_entry = cfg.regime_entry_thresholds[0]
        bear_entry = cfg.regime_entry_thresholds[2]
        assert bull_entry < bear_entry  # Bull: 1.5, Bear: 2.5


class TestMetaSignalModel:
    """Test MetaSignalModel functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = MetaSignalModel()
        assert model.weights is not None
        assert model.signal_history is not None

    def test_custom_config(self):
        """Test initialization with custom config."""
        cfg = MetaSignalConfig(use_learned_weights=True)
        model = MetaSignalModel(config=cfg)
        assert model.cfg.use_learned_weights is True

    def test_combine_signals_weighted(self):
        """Test weighted signal combination."""
        model = MetaSignalModel()
        signals = {
            "zscore": 3.0,
            "momentum": 0.01,
            "zscore_momentum": 1.5,
        }
        combined = model.combine_signals(signals, method="weighted")
        assert -1.0 <= combined <= 1.0
        # With default weights and these signals, result should be positive
        assert combined > 0

    def test_combine_signals_simple_average(self):
        """Test simple average signal combination."""
        model = MetaSignalModel()
        signals = {
            "zscore": 3.0,
            "momentum": 0.01,
            "zscore_momentum": 1.5,
        }
        combined = model.combine_signals(signals, method="simple_average")
        assert -1.0 <= combined <= 1.0
        assert combined > 0

    def test_combine_signals_empty(self):
        """Test combination with no signals."""
        model = MetaSignalModel()
        signals = {}
        combined = model.combine_signals(signals, method="weighted")
        assert combined == 0.0

    def test_combine_signals_clipping(self):
        """Test that signals are clipped to [-1, 1]."""
        model = MetaSignalModel()
        # Extreme signals that would exceed bounds
        signals = {
            "zscore": 100.0,
            "momentum": 1.0,
            "zscore_momentum": 50.0,
        }
        combined = model.combine_signals(signals, method="weighted")
        assert -1.0 <= combined <= 1.0

    def test_apply_regime_scaling(self):
        """Test regime-dependent signal scaling."""
        model = MetaSignalModel()
        base_signal = 0.8

        # Bull regime: full strength
        bull_scaled = model.apply_regime_scaling(base_signal, current_regime=0)
        assert bull_scaled == pytest.approx(0.8)

        # Bear regime: half strength
        bear_scaled = model.apply_regime_scaling(base_signal, current_regime=2)
        assert bear_scaled == pytest.approx(0.4)

        # Crisis regime: minimal strength
        crisis_scaled = model.apply_regime_scaling(base_signal, current_regime=3)
        assert crisis_scaled == pytest.approx(0.16)

    def test_get_entry_threshold(self):
        """Test regime-dependent entry thresholds."""
        model = MetaSignalModel()

        # Bull: tight entry
        bull_entry = model.get_entry_threshold(current_regime=0)
        assert bull_entry == pytest.approx(1.5)

        # Bear: loose entry
        bear_entry = model.get_entry_threshold(current_regime=2)
        assert bear_entry == pytest.approx(2.5)
        assert bear_entry > bull_entry

    def test_get_exit_threshold(self):
        """Test regime-dependent exit thresholds."""
        model = MetaSignalModel()

        # Bull: exit close to mean
        bull_exit = model.get_exit_threshold(current_regime=0)
        assert bull_exit == pytest.approx(0.3)

        # Bear: exit at partial reversion
        bear_exit = model.get_exit_threshold(current_regime=2)
        assert bear_exit == pytest.approx(0.8)
        assert bear_exit > bull_exit

    def test_get_stop_loss_threshold(self):
        """Test regime-dependent stop-loss thresholds."""
        model = MetaSignalModel()

        # Bull: loose stop
        bull_stop = model.get_stop_loss_threshold(current_regime=0)
        assert bull_stop == pytest.approx(3.5)

        # Bear: tight stop
        bear_stop = model.get_stop_loss_threshold(current_regime=2)
        assert bear_stop == pytest.approx(3.0)

        # Crisis: very tight stop
        crisis_stop = model.get_stop_loss_threshold(current_regime=3)
        assert crisis_stop == pytest.approx(2.5)

    def test_get_position_scale(self):
        """Test regime-dependent position size scaling."""
        model = MetaSignalModel()

        # Bull: full size
        bull_scale = model.get_position_scale(current_regime=0)
        assert bull_scale == pytest.approx(1.0)

        # Bear: half size
        bear_scale = model.get_position_scale(current_regime=2)
        assert bear_scale == pytest.approx(0.5)

        # Crisis: minimal size
        crisis_scale = model.get_position_scale(current_regime=3)
        assert crisis_scale == pytest.approx(0.1)

    def test_record_outcome_no_learning(self):
        """Test recording signals without triggering learning."""
        cfg = MetaSignalConfig(use_learned_weights=False)
        model = MetaSignalModel(config=cfg)
        initial_weights = dict(model.weights)

        signals = {"zscore": 1.5, "momentum": 0.005}
        for _ in range(15):
            model.record_outcome(signals, 100.0)

        # Weights should not change when learning is disabled
        assert model.weights == initial_weights

    def test_record_outcome_with_learning_triggered(self):
        """Test that learning adjusts weights on positive outcomes."""
        cfg = MetaSignalConfig(use_learned_weights=True)
        model = MetaSignalModel(config=cfg)
        initial_weights = dict(model.weights)

        signals = {"zscore": 1.5, "momentum": 0.005}
        # Record 10 positive trades
        for _ in range(10):
            model.record_outcome(signals, 100.0)

        # After 10 positive trades, weights should shift (zscore up)
        assert model.weights["zscore"] > initial_weights["zscore"]
        assert model.weights["momentum"] < initial_weights["momentum"]

    def test_weight_history_tracking(self):
        """Test that weight history is tracked during learning."""
        cfg = MetaSignalConfig(use_learned_weights=True)
        model = MetaSignalModel(config=cfg)
        initial_count = len(model.weight_history)

        signals = {"zscore": 1.5, "momentum": 0.005}
        # Record 10 positive trades
        for _ in range(10):
            model.record_outcome(signals, 100.0)

        # Weight history should be updated
        assert len(model.weight_history) > initial_count

    def test_regime_configuration_override(self):
        """Test that custom regime configurations override defaults."""
        custom_entry = {0: 1.2, 1: 2.0, 2: 3.0, 3: 5.0}
        cfg = MetaSignalConfig(regime_entry_thresholds=custom_entry)
        model = MetaSignalModel(config=cfg)

        assert model.get_entry_threshold(0) == pytest.approx(1.2)
        assert model.get_entry_threshold(2) == pytest.approx(3.0)

    def test_all_regimes_have_defaults(self):
        """Test that all 4 regimes have defined values."""
        model = MetaSignalModel()
        for regime in [0, 1, 2, 3]:
            entry = model.get_entry_threshold(regime)
            exit_thr = model.get_exit_threshold(regime)
            stop = model.get_stop_loss_threshold(regime)
            scale = model.get_position_scale(regime)
            assert entry > 0
            assert exit_thr > 0
            assert stop > 0
            assert 0 <= scale <= 1.0

    def test_weights_sum_positive(self):
        """Test that weight sum is always positive for combination."""
        model = MetaSignalModel()
        w_sum = sum(model.weights.values())
        assert w_sum > 0

    def test_signal_scaling_preserves_sign(self):
        """Test that regime scaling preserves signal sign."""
        model = MetaSignalModel()

        # Positive signal
        pos_result = model.apply_regime_scaling(0.5, current_regime=2)
        assert pos_result > 0

        # Negative signal
        neg_result = model.apply_regime_scaling(-0.5, current_regime=2)
        assert neg_result < 0

        # Zero signal
        zero_result = model.apply_regime_scaling(0.0, current_regime=2)
        assert zero_result == 0.0

    def test_crisis_regime_blocks_entries(self):
        """Test that crisis regime effectively blocks entries."""
        model = MetaSignalModel()
        base_signal = 1.0
        crisis_scale = model.get_position_scale(current_regime=3)

        # Position scale in crisis should be very small
        assert crisis_scale <= 0.2
        
        # Scaled signal should be small
        crisis_scaled = model.apply_regime_scaling(base_signal, current_regime=3)
        assert crisis_scaled <= 0.2

    def test_neutral_regime_middle_ground(self):
        """Test that neutral regime is between bull and bear."""
        model = MetaSignalModel()

        bull_pos = model.get_position_scale(0)
        neutral_pos = model.get_position_scale(1)
        bear_pos = model.get_position_scale(2)

        assert bear_pos < neutral_pos < bull_pos

    def test_threshold_realism(self):
        """Test that thresholds are in reasonable ranges for z-scores."""
        model = MetaSignalModel()

        for regime in [0, 1, 2, 3]:
            entry = model.get_entry_threshold(current_regime=regime)
            exit_thr = model.get_exit_threshold(current_regime=regime)
            stop = model.get_stop_loss_threshold(current_regime=regime)

            # Entry threshold should be 1-5 std devs
            assert 1.0 <= entry <= 5.0
            # Exit should be < entry
            assert exit_thr < entry
            # Stop should be > exit
            assert stop > exit_thr
