"""Meta-signal model and regime-aware configuration.

Combines ensemble signals into final trading strength with optional learned weights.
Regime-aware sizing and thresholds.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MetaSignalConfig:
    """Configuration for meta-signal model and regime adaptation.
    
    Attributes
    ----------
    use_learned_weights : bool
        If True, learn signal weights over time. Else use fixed weights.
    fixed_weights : dict
        Default weights for signal combination (if not learning).
    regime_strength_scale : dict
        Multiply final signal strength by regime-dependent factor.
    regime_entry_thresholds : dict
        Entry z-score threshold per regime (lower = more aggressive).
    regime_exit_thresholds : dict
        Exit z-score threshold per regime.
    regime_stop_loss : dict
        Stop-loss z-score threshold per regime.
    regime_position_scale : dict
        Multiply position size by regime-dependent factor.
        """
    use_learned_weights: bool = False
    fixed_weights: Dict[str, float] = field(default_factory=lambda: {
        "zscore": 0.6,
        "momentum": 0.2,
        "zscore_momentum": 0.2,
    })
    regime_strength_scale: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # Bull: full size
        1: 0.8,   # Neutral
        2: 0.5,   # Bear: half size
        3: 0.2,   # Crisis: minimal
    })
    regime_entry_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 1.5,   # Bull: tighter
        1: 2.0,   # Neutral
        2: 2.5,   # Bear: wider
        3: 4.0,   # Crisis: block
    })
    regime_exit_thresholds: Dict[int, float] = field(default_factory=lambda: {
        0: 0.3,   # Bull: exit close
        1: 0.5,   # Neutral
        2: 0.8,   # Bear: exit early
        3: 1.0,   # Crisis: exit any reversion
    })
    regime_stop_loss: Dict[int, float] = field(default_factory=lambda: {
        0: 3.5,   # Bull: loose
        1: 3.5,   # Neutral
        2: 3.0,   # Bear: tighter
        3: 2.5,   # Crisis: tight
    })
    regime_position_scale: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # Bull: full size
        1: 0.75,  # Neutral
        2: 0.5,   # Bear: half
        3: 0.1,   # Crisis: minimal
    })


class MetaSignalModel:
    """Meta-model for combining ensemble signals via weighted average or learned weights.
    
    If use_learned_weights=True, adapts weights based on recent signal performance.
    Otherwise uses fixed weights.
    """

    def __init__(self, config: Optional[MetaSignalConfig] = None):
        self.cfg = config or MetaSignalConfig()
        self.weights = dict(self.cfg.fixed_weights)
        
        # History for learning (if enabled)
        self.signal_history = []  # list of (signals_dict, outcome)
        self.weight_history = [dict(self.weights)]

    def combine_signals(
        self,
        ensemble_signals: Dict[str, float],
        method: str = "weighted",
    ) -> float:
        """Combine ensemble signals into a single strength value.
        
        Parameters
        ----------
        ensemble_signals : dict
            Output from SignalEnsemble.compute_signals()
        method : str
            'weighted' for fixed/learned weights, 'simple_average' for unweighted
            
        Returns
        -------
        float: combined signal in [-1, 1] range (approx)
        """
        if method == "simple_average":
            # Simple average of normalized signals
            z_norm = np.clip(ensemble_signals.get("zscore", 0.0) / 3.0, -1.0, 1.0)
            mom_norm = np.clip(ensemble_signals.get("momentum", 0.0) / 0.01, -1.0, 1.0)
            return (z_norm + mom_norm) / 2.0
        
        # Weighted combination
        w_total = sum(self.weights.values())
        if w_total <= 0:
            return 0.0
        
        z_norm = np.clip(ensemble_signals.get("zscore", 0.0) / 3.0, -1.0, 1.0)
        mom_norm = np.clip(ensemble_signals.get("momentum", 0.0) / 0.01, -1.0, 1.0)
        zmom_norm = np.clip(ensemble_signals.get("zscore_momentum", 0.0) / 3.0, -1.0, 1.0)
        
        combined = (
            self.weights.get("zscore", 0.0) * z_norm
            + self.weights.get("momentum", 0.0) * mom_norm
            + self.weights.get("zscore_momentum", 0.0) * zmom_norm
        ) / w_total
        
        return float(np.clip(combined, -1.0, 1.0))

    def apply_regime_scaling(
        self,
        base_signal: float,
        current_regime: int,
    ) -> float:
        """Apply regime-dependent scaling to base signal.
        
        Parameters
        ----------
        base_signal : float
            Pre-regime signal strength
        current_regime : int
            Current regime label
            
        Returns
        -------
        float: regime-scaled signal
        """
        scale = self.cfg.regime_strength_scale.get(current_regime, 1.0)
        return base_signal * scale

    def get_entry_threshold(self, current_regime: int) -> float:
        """Get entry z-score threshold for current regime."""
        return self.cfg.regime_entry_thresholds.get(current_regime, 2.0)

    def get_exit_threshold(self, current_regime: int) -> float:
        """Get exit z-score threshold for current regime."""
        return self.cfg.regime_exit_thresholds.get(current_regime, 0.5)

    def get_stop_loss_threshold(self, current_regime: int) -> float:
        """Get stop-loss z-score threshold for current regime."""
        return self.cfg.regime_stop_loss.get(current_regime, 3.5)

    def get_position_scale(self, current_regime: int) -> float:
        """Get position size scale factor for current regime."""
        return self.cfg.regime_position_scale.get(current_regime, 1.0)

    def record_outcome(self, signals: Dict[str, float], profit_loss: float) -> None:
        """Record signal performance for potential future learning.
        
        Parameters
        ----------
        signals : dict
            Ensemble signals
        profit_loss : float
            Realized P&L from the trade
        """
        if self.cfg.use_learned_weights:
            self.signal_history.append((signals, profit_loss))
            # Simple adaptation: if recent trades positive, boost primary signal
            # (more sophisticated approaches would use regression)
            if len(self.signal_history) >= 10:
                recent_pl = sum(pl for _, pl in self.signal_history[-10:])
                if recent_pl > 0:
                    # Slightly increase zscore weight
                    self.weights["zscore"] = min(0.8, self.weights["zscore"] + 0.01)
                    self.weights["momentum"] = max(0.1, self.weights["momentum"] - 0.005)
                    self.weight_history.append(dict(self.weights))
