"""Strategy wrapper: adapts PairsTradingStrategy into the event-driven engine.

Responsibilities:
    - Maintains a rolling price history window (per-ticker deques)
    - On each MarketEvent: recomputes spread z-scores for all registered pairs
    - Manages state machine (flat / long_spread / short_spread) per pair
    - Optionally gates entries on regime label (HMM / Vol / KMeans detector)
    - Emits SignalEvents consumed by the BacktestEngine's position sizer

Regime-adaptive sizing:
    If a regime detector is supplied, `strength` on emitted signals is scaled by
    `regime_size_map[current_regime]` (default: 1.0 for low/mid vol, 0.5 for
    high vol, 0.0 for crisis to skip new entries).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

from .events import MarketEvent, SignalEvent
from .portfolio import Portfolio

logger = logging.getLogger(__name__)

# Default: per-regime target-notional multipliers
DEFAULT_REGIME_SIZE: dict[int, float] = {
    0: 1.0,    # Bull / Low-Vol  — full size
    1: 0.75,   # Neutral         — reduce slightly
    2: 0.4,    # Bear / High-Vol — half size
    3: 0.0,    # Crisis          — no new entries
}


class PairConfig:
    """Data for a single registered pair."""
    def __init__(self, ticker1: str, ticker2: str, hedge_ratio: float,
                 entry_z: float, exit_z: float, stop_z: float):
        self.ticker1      = ticker1
        self.ticker2      = ticker2
        self.hedge_ratio  = hedge_ratio
        self.entry_z      = entry_z
        self.exit_z       = exit_z
        self.stop_z       = stop_z
        self.pair_id      = f"{ticker1}/{ticker2}"
        self.position: int = 0   # -1 / 0 / +1


class PairsBacktestStrategy:
    """Event-driven wrapper for multiple simultaneous pairs.

    Parameters
    ----------
    pairs : list of dicts with keys: ticker1, ticker2, hedge_ratio
        Use the output of PairsSelector.find_pairs().
    zscore_window : int
        Rolling window for z-score.
    entry_z / exit_z / stop_z : float
    regime_detector : optional fitted BaseRegimeDetector
        If supplied, predicts regime each bar (on the `regime_ticker` price
        history) and scales position strength accordingly.
    regime_ticker : str
        Ticker used to drive regime detection (default 'SPY' or first ticker).
    regime_size_map : dict[int, float]
        Multipliers per regime label.
    warmup_bars : int
        Minimum price history before emitting any signals.
    """

    def __init__(
        self,
        pairs: List[dict],
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float  = 0.5,
        stop_z: float  = 3.5,
        regime_detector=None,
        regime_ticker: Optional[str] = None,
        regime_size_map: Optional[dict] = None,
        warmup_bars: int = 60,
        strategy_id: str = "pairs",
    ):
        self.zscore_window  = zscore_window
        self.warmup_bars    = warmup_bars
        self.strategy_id    = strategy_id

        # Pair configs
        self._pairs: List[PairConfig] = []
        for p in pairs:
            self._pairs.append(PairConfig(
                ticker1     = p["ticker1"],
                ticker2     = p["ticker2"],
                hedge_ratio = p["hedge_ratio"],
                entry_z     = p.get("entry_z", entry_z),
                exit_z      = p.get("exit_z",  exit_z),
                stop_z      = p.get("stop_z",  stop_z),
            ))

        # Collect all unique tickers
        all_tickers: set[str] = set()
        for pc in self._pairs:
            all_tickers.update([pc.ticker1, pc.ticker2])
        self._tickers = list(all_tickers)

        # Price history buffers (enough for z-score + regime window)
        buf_size = max(zscore_window * 3, 252)
        self._price_buf: Dict[str, deque] = {
            t: deque(maxlen=buf_size) for t in self._tickers
        }

        # Regime
        self._regime_detector  = regime_detector
        self._regime_ticker    = regime_ticker or (self._tickers[0] if self._tickers else "SPY")
        self._regime_size_map  = regime_size_map or DEFAULT_REGIME_SIZE
        self._regime_buf: deque = deque(maxlen=max(zscore_window * 3, 252))
        self._current_regime: int = 1   # start neutral

        self._bar_count = 0

    # -----------------------------------------------------------------------
    # Called once per bar by the BacktestEngine
    # -----------------------------------------------------------------------

    def on_market_event(
        self,
        event: MarketEvent,
        portfolio: Portfolio,
    ) -> List[SignalEvent]:

        # 1. Update price buffers
        for t in self._tickers:
            p = event.prices.get(t)
            if p is not None and np.isfinite(p):
                self._price_buf[t].append(p)

        self._bar_count += 1
        if self._bar_count < self.warmup_bars:
            return []

        # 2. Update regime if detector is attached
        if self._regime_detector is not None:
            self._update_regime(event)

        # 3. Generate signals per pair
        signals = []
        for pc in self._pairs:
            sig = self._evaluate_pair(event, pc)
            if sig is not None:
                signals.append(sig)

        return signals

    # -----------------------------------------------------------------------
    # Pair evaluation
    # -----------------------------------------------------------------------

    def _evaluate_pair(self, event: MarketEvent, pc: PairConfig) -> Optional[SignalEvent]:
        buf1 = list(self._price_buf.get(pc.ticker1, []))
        buf2 = list(self._price_buf.get(pc.ticker2, []))
        n    = min(len(buf1), len(buf2))
        if n < self.zscore_window // 2:
            return None

        s1 = np.array(buf1[-n:], dtype=float)
        s2 = np.array(buf2[-n:], dtype=float)
        spread = s1 - pc.hedge_ratio * s2

        win = min(self.zscore_window, n)
        mu  = spread[-win:].mean()
        sig = spread[-win:].std()
        if sig < 1e-10:
            return None

        z = (spread[-1] - mu) / sig

        # State machine
        new_dir = None
        cur     = pc.position

        # Exit / stop
        if cur != 0:
            if abs(z) < pc.exit_z or abs(z) > pc.stop_z:
                new_dir = "flat"

        if new_dir is None:
            # Entry
            if cur == 0:
                regime_scale = self._regime_size_map.get(self._current_regime, 1.0)
                if regime_scale == 0.0:
                    return None      # regime blocks new entries

                if z < -pc.entry_z:
                    new_dir = "long_spread"
                elif z > pc.entry_z:
                    new_dir = "short_spread"

        if new_dir is None and cur == pc.position:
            # Re-emit current direction so position sizer can rebalance if equity shifted
            return None

        if new_dir is not None:
            if new_dir == "flat":
                pc.position = 0
            elif new_dir == "long_spread":
                pc.position = 1
            elif new_dir == "short_spread":
                pc.position = -1

            regime_scale = self._regime_size_map.get(self._current_regime, 1.0) if new_dir != "flat" else 1.0
            return SignalEvent(
                date        = event.date,
                ticker1     = pc.ticker1,
                ticker2     = pc.ticker2,
                direction   = new_dir,
                strength    = regime_scale,
                spread_zscore = z,
                hedge_ratio = pc.hedge_ratio,
                strategy_id = self.strategy_id,
            )
        return None

    # -----------------------------------------------------------------------
    # Regime
    # -----------------------------------------------------------------------

    def _update_regime(self, event: MarketEvent) -> None:
        """Refit/predict regime on the latest price window."""
        rt  = self._regime_ticker
        buf = list(self._price_buf.get(rt, []))
        if len(buf) < self.zscore_window:
            return

        # Build a minimal feature df for the detector from the price buffer
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from features.featurize import compute_standard_features

            prices = pd.Series(buf, name="close")
            dates  = pd.date_range(end=event.date, periods=len(buf), freq="B")
            mini_df = pd.DataFrame({"Date": dates, "close": prices.values})
            feat = compute_standard_features(mini_df)

            if "rv_20" not in feat.columns or feat["rv_20"].dropna().empty:
                return

            feat = feat.set_index("Date") if "Date" in feat.columns else feat
            labels = self._regime_detector.predict(feat)
            if len(labels) > 0:
                self._current_regime = int(labels.iloc[-1])
        except Exception as e:
            logger.debug(f"Regime update skipped: {e}")
