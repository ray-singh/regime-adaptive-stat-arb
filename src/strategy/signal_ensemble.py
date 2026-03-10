"""Signal ensemble for pairs trading.

Combines multiple signal types (z-score, momentum, mean-reversion strength)
to generate robust trading decisions.

Signals:
    1. Z-score signal: standard deviation-based mean-reversion
    2. Momentum signal: rate of change of spread
    3. Half-life signal: mean-reversion speed (if available)
    4. Volatility-adjusted signal: scale by regime volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


class SignalEnsemble:
    """Multimodal signal generator for pairs.

    Parameters
    ----------
    zscore_window : int
        Window for z-score computation.
    momentum_window : int
        Window for momentum computation (first difference).
    min_vol : float
        Minimum volatility to avoid division by zero.
    """

    def __init__(
        self,
        zscore_window: int = 60,
        momentum_window: int = 5,
        min_vol: float = 1e-6,
    ):
        self.zscore_window = zscore_window
        self.momentum_window = momentum_window
        self.min_vol = min_vol

    def compute_signals(
        self,
        spread: np.ndarray,
    ) -> Dict[str, float]:
        """Compute multimodal signals from a spread time series.

        Parameters
        ----------
        spread : array of shape (n,)
            Time series of spread values.

        Returns
        -------
        dict with keys:
            - 'zscore': standard deviation-based z-score
            - 'momentum': rate of change of spread
            - 'zscore_momentum': z-score of momentum (carry)
            - 'spread_level': normalized spread level
        """
        n = len(spread)
        if n < 2:
            return {
                "zscore": 0.0,
                "momentum": 0.0,
                "zscore_momentum": 0.0,
                "spread_level": 0.0,
            }

        # Z-score signal — use available window size
        win_z = min(self.zscore_window, n)
        mu = spread[-win_z:].mean()
        sig = spread[-win_z:].std()
        sig = max(sig, self.min_vol)
        zscore = (spread[-1] - mu) / sig

        # Momentum signal (first difference)
        if n < 2:
            momentum = 0.0
        else:
            look_back = min(self.momentum_window, n - 1)
            delta = np.diff(spread[-look_back - 1 :])
            momentum = delta.mean()

        # Z-score of momentum (carry signal)
        if n < self.momentum_window + 5:
            zscore_momentum = 0.0
        else:
            deltas = np.diff(spread[-self.momentum_window - 5 :])
            mom_mu = deltas.mean()
            mom_sig = max(deltas.std(), self.min_vol)
            zscore_momentum = (momentum - mom_mu) / mom_sig

        # Normalized spread level (rank relative to recent range)
        min_sp = spread[-win_z:].min()
        max_sp = spread[-win_z:].max()
        spread_range = max_sp - min_sp
        spread_level = (
            (spread[-1] - min_sp) / spread_range if spread_range > self.min_vol else 0.5
        )

        return {
            "zscore": float(zscore),
            "momentum": float(momentum),
            "zscore_momentum": float(zscore_momentum),
            "spread_level": float(spread_level),
        }

    def signal_strength(
        self,
        signals: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Combine signals into a single strength value.

        Parameters
        ----------
        signals : dict
            Output from compute_signals()
        weights : dict, optional
            Weights for each signal component.
            If None, uses equal weights.

        Returns
        -------
        float: combined signal strength in [-1, 1] range (approx)
            Positive = long signal
            Negative = short signal
        """
        if weights is None:
            weights = {
                "zscore": 0.6,  # primary signal
                "momentum": 0.2,  # secondary signal
                "zscore_momentum": 0.2,  # carry signal
            }

        # Normalize momentum and zscore_momentum to rough [-1, 1] range
        # (z-score can exceed this, but we clip for combination)
        z_norm = np.clip(signals.get("zscore", 0.0) / 3.0, -1.0, 1.0)
        mom_norm = np.clip(
            signals.get("momentum", 0.0) / 0.01, -1.0, 1.0
        )  # typical spread momentum ~0.01
        zmom_norm = np.clip(signals.get("zscore_momentum", 0.0) / 3.0, -1.0, 1.0)

        w_total = sum(weights.values())
        if w_total <= 0:
            return 0.0

        combined = (
            weights.get("zscore", 0.0) * z_norm
            + weights.get("momentum", 0.0) * mom_norm
            + weights.get("zscore_momentum", 0.0) * zmom_norm
        ) / w_total

        return float(np.clip(combined, -1.0, 1.0))

    def entry_signal(
        self,
        signals: Dict[str, float],
        entry_threshold: float = 2.0,
        momentum_confirm: bool = True,
    ) -> Optional[str]:
        """Generate entry direction based on ensemble signals.

        Parameters
        ----------
        signals : dict
            Output from compute_signals()
        entry_threshold : float
            Z-score threshold for entry
        momentum_confirm : bool
            If True, also require positive momentum in the entry direction

        Returns
        -------
        None, 'long_spread', or 'short_spread'
        """
        zscore = signals.get("zscore", 0.0)
        momentum = signals.get("momentum", 0.0)

        if zscore < -entry_threshold:
            if not momentum_confirm or momentum < 0.0:
                return "long_spread"  # spread is negative, likely to mean-revert up
        elif zscore > entry_threshold:
            if not momentum_confirm or momentum > 0.0:
                return "short_spread"  # spread is positive, likely to mean-revert down

        return None

    def exit_signal(
        self,
        signals: Dict[str, float],
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.5,
    ) -> Optional[str]:
        """Generate exit direction based on ensemble signals.

        Parameters
        ----------
        signals : dict
            Output from compute_signals()
        exit_threshold : float
            Z-score threshold for profit-taking exit
        stop_loss_threshold : float
            Z-score threshold for stop-loss

        Returns
        -------
        None or 'flat'
        """
        zscore = abs(signals.get("zscore", 0.0))

        if zscore < exit_threshold or zscore > stop_loss_threshold:
            return "flat"

        return None
