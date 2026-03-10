"""Unit tests for Kalman filter hedge ratio estimation."""
import pytest
import numpy as np
from strategy.kalman_hedge import KalmanHedge


class TestKalmanHedgeBasics:
    """Test basic Kalman filter mechanics."""

    def test_initialization(self):
        kf = KalmanHedge(initial_hedge=1.0, initial_uncertainty=1.0)
        assert kf.hedge == 1.0
        assert kf.uncertainty == 1.0

    def test_single_update_with_perfect_spread(self):
        """If spread is at 0 and prices match the hedge, hedge should not change."""
        kf = KalmanHedge(initial_hedge=1.0, process_variance=0.0001, measurement_variance=0.01)
        # If p1 = 100 and p2 = 100, spread = 0 → hedge should remain ~1.0
        hedge, unc = kf.update(100.0, 100.0)
        assert abs(hedge - 1.0) < 0.01  # hedge should stay close to 1.0

    def test_update_shifts_hedge_toward_observation(self):
        """If spread is positive, hedge should increase (to shrink the spread)."""
        kf = KalmanHedge(initial_hedge=1.0, process_variance=0.01, measurement_variance=0.01)
        # p1 = 110, p2 = 100, spread = 10 when hedge = 1.0
        # To reduce spread, we need hedge > 1.0 so that hedge*p2 gets larger
        hedge, _ = kf.update(110.0, 100.0)
        assert hedge > 1.0  # hedge increases when spread is positive

    def test_uncertainty_converges(self):
        """Repeated observations should reduce uncertainty."""
        kf = KalmanHedge(initial_hedge=1.0, initial_uncertainty=10.0, process_variance=0.0001)
        unc_history = [kf.uncertainty]
        for _ in range(50):
            _, unc = kf.update(100.0, 100.0)
            unc_history.append(unc)
        # Uncertainty should decrease over time
        assert unc_history[-1] < unc_history[0]

    def test_infinity_and_nan_handling(self):
        """Filter should handle inf/nan prices gracefully."""
        kf = KalmanHedge(initial_hedge=1.0)
        hedge_before, unc_before = kf.hedge, kf.uncertainty
        # Try to update with inf
        hedge, unc = kf.update(float("inf"), 100.0)
        assert hedge == hedge_before  # state unchanged
        assert unc == unc_before
        # Try with nan
        hedge, unc = kf.update(100.0, float("nan"))
        assert hedge == hedge_before
        assert unc == unc_before
        # Try with zero price2
        hedge, unc = kf.update(100.0, 0.0)
        assert hedge == hedge_before
        assert unc == unc_before

    def test_zero_price2_handling(self):
        """Division by zero should be avoided."""
        kf = KalmanHedge(initial_hedge=1.0)
        # Should not raise; should return unchanged state
        hedge, unc = kf.update(100.0, 0.0)
        assert hedge == 1.0
        assert unc == kf.uncertainty


class TestKalmanHedgeTracking:
    """Test hedge ratio tracking in regime-like scenarios."""

    def test_hedge_drifts_with_changing_relationship(self):
        """If price ratio shifts, hedge should track it."""
        kf = KalmanHedge(initial_hedge=1.0, process_variance=0.001, measurement_variance=0.01)
        # Initially p1 ≈ p2, so hedge ≈ 1.0
        for i in range(20):
            kf.update(100.0 + i, 100.0)
        hedge_early = kf.hedge

        # Now p1 ≈ 2 * p2, so hedge should move toward 2.0
        for i in range(30):
            kf.update(200.0, 100.0)
        hedge_late = kf.hedge

        # Hedge should have shifted towards 2.0
        assert hedge_late > hedge_early
        assert hedge_late > 1.5  # should be closer to 2.0

    def test_concurrent_pair_filters(self):
        """Two filters should maintain roughly independent state if price ratio is clear."""
        kf1 = KalmanHedge(initial_hedge=1.0, process_variance=0.001, measurement_variance=0.1)
        kf2 = KalmanHedge(initial_hedge=2.0, process_variance=0.001, measurement_variance=0.1)

        # With perfect 1:1 price ratio, both should converge to 1.0 (expected hedge)
        for _ in range(10):
            kf1.update(100.0, 100.0)
            kf2.update(100.0, 100.0)

        # Both should have converged to ~1.0 (true hedge for equal-priced assets)
        assert abs(kf1.hedge - 1.0) < 0.05
        assert abs(kf2.hedge - 1.0) < 0.05


class TestKalmanHedgeStateManagement:
    """Test save/restore and reset."""

    def test_get_state(self):
        kf = KalmanHedge(initial_hedge=1.5, initial_uncertainty=2.0)
        state = kf.get_state()
        assert state["hedge"] == 1.5
        assert state["uncertainty"] == 2.0

    def test_set_state(self):
        kf = KalmanHedge()
        kf.set_state(hedge=1.5, uncertainty=3.0)
        assert kf.hedge == 1.5
        assert kf.uncertainty == 3.0

    def test_reset(self):
        kf = KalmanHedge(initial_hedge=1.0)
        kf.update(110.0, 100.0)  # drift hedge
        assert kf.hedge != 1.0
        kf.reset(initial_hedge=1.0)
        assert kf.hedge == 1.0
        assert kf.uncertainty == 1.0

    def test_history_tracking(self):
        kf = KalmanHedge(initial_hedge=1.0)
        assert len(kf.hedge_history) == 1
        kf.update(100.0, 100.0)
        assert len(kf.hedge_history) == 2
        kf.update(110.0, 100.0)
        assert len(kf.hedge_history) == 3


class TestKalmanHedgeVarianceEffects:
    """Test process and measurement variance effects."""

    def test_high_process_variance_responsive(self):
        """Higher process variance = faster hedge adaptation to regime change."""
        kf_slow = KalmanHedge(initial_hedge=1.0, process_variance=0.00001, measurement_variance=0.01)
        kf_fast = KalmanHedge(initial_hedge=1.0, process_variance=0.01, measurement_variance=0.01)

        # Sudden regime change: p1 ≈ 2*p2
        for _ in range(5):
            kf_slow.update(200.0, 100.0)
            kf_fast.update(200.0, 100.0)

        # Fast should be closer to 2.0 (faster adaptation to new regime)
        dist_slow = abs(kf_slow.hedge - 2.0)
        dist_fast = abs(kf_fast.hedge - 2.0)
        assert dist_fast < dist_slow  # fast tracks better

    def test_high_measurement_variance_trusts_less(self):
        """Higher measurement variance = less trust in observations."""
        kf_trusts = KalmanHedge(initial_hedge=1.0, measurement_variance=0.001)
        kf_skeptic = KalmanHedge(initial_hedge=1.0, measurement_variance=10.0)

        kf_trusts.update(200.0, 100.0)
        kf_skeptic.update(200.0, 100.0)

        # Trusts should have updated more
        assert abs(kf_trusts.hedge - 2.0) < abs(kf_skeptic.hedge - 2.0)
