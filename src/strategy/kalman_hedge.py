"""Kalman filter for dynamic hedge ratio estimation.

The Kalman filter estimates the time-varying hedge ratio between two
cointegrated assets. As market regimes shift, the optimal hedge ratio
can drift; Kalman tracks these changes online.

Usage:
    kf = KalmanHedge(initial_hedge=1.0, process_var=0.0001, meas_var=0.01)
    for p1, p2 in price_pairs:
        hedge, uncertainty = kf.update(p1, p2)
        spread = p1 - hedge * p2
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class KalmanHedge:
    """1D Kalman filter for online hedge ratio estimation.

    State: hedge_ratio (scalar)
    Observation: spread = p1 - hedge_ratio * p2

    Parameters
    ----------
    initial_hedge : float
        Starting estimate of hedge ratio (typically ~1.0 for pairs).
    initial_uncertainty : float
        Initial state covariance (uncertainty in hedge ratio estimate).
    process_variance : float
        How much the hedge ratio is allowed to drift between observations.
        Larger = more responsive to regime changes.
    measurement_variance : float
        Noise level in the observed spread. Affects trust in new price data.
        """

    def __init__(
        self,
        initial_hedge: float = 1.0,
        initial_uncertainty: float = 1.0,
        process_variance: float = 0.0001,
        measurement_variance: float = 0.01,
    ):
        self.hedge = initial_hedge
        self.uncertainty = initial_uncertainty
        self.process_var = process_variance
        self.meas_var = measurement_variance

        # History for diagnostics
        self.hedge_history = [initial_hedge]
        self.uncertainty_history = [initial_uncertainty]

    def update(self, price1: float, price2: float) -> Tuple[float, float]:
        """Update hedge ratio estimate given new prices.

        Returns (hedge_ratio, uncertainty) for the current step.
        """
        if not np.isfinite(price1) or not np.isfinite(price2) or price2 == 0:
            return self.hedge, self.uncertainty

        # --- PREDICT STEP ---
        # Prior: hedge ratio drifts slightly (random walk with small variance)
        predicted_hedge = self.hedge
        predicted_uncertainty = self.uncertainty + self.process_var

        # --- UPDATE STEP ---
        # Observation: spread = price1 - predicted_hedge * price2
        # We want the spread to be 0 (or stationary around its mean)
        # Innovation: observed spread
        spread = price1 - predicted_hedge * price2

        # Measurement sensitivity: H = price2
        # (spread changes by price2 for a unit change in hedge)
        H = price2

        # Kalman gain: how much to trust the new observation
        S = predicted_uncertainty * H * H + self.meas_var  # innovation covariance
        if abs(S) < 1e-12:
            K = 0.0
        else:
            K = predicted_uncertainty * H / S

        # Update hedge and uncertainty
        self.hedge = predicted_hedge + K * spread
        self.uncertainty = (1.0 - K * H) * predicted_uncertainty

        # Ensure uncertainty stays positive and bounded
        self.uncertainty = np.clip(self.uncertainty, 1e-6, 100.0)
        self.hedge_history.append(self.hedge)
        self.uncertainty_history.append(self.uncertainty)

        return self.hedge, self.uncertainty

    def get_state(self) -> dict:
        """Return current filter state."""
        return {
            "hedge": self.hedge,
            "uncertainty": self.uncertainty,
            "process_var": self.process_var,
            "meas_var": self.meas_var,
        }

    def set_state(self, hedge: float, uncertainty: float) -> None:
        """Manually set filter state (e.g., from saved model)."""
        self.hedge = float(hedge)
        self.uncertainty = float(uncertainty)

    def reset(self, initial_hedge: float = 1.0) -> None:
        """Reset filter to a clean state."""
        self.hedge = initial_hedge
        self.uncertainty = 1.0
        self.hedge_history = [initial_hedge]
        self.uncertainty_history = [1.0]
