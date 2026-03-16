"""Strategy wrapper: adapts PairsTradingStrategy into the event-driven engine.

Responsibilities:
    - Maintains a rolling price history window (per-ticker deques)
    - On each MarketEvent: recomputes spread z-scores for all registered pairs
    - Manages state machine (flat / long_spread / short_spread) per pair
    - Optionally gates entries on regime label (HMM / Vol / KMeans detector)
    - Emits SignalEvents consumed by the BacktestEngine's position sizer
    - Periodically re-selects pairs via PairReSelector (if configured)

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
from strategy.kalman_hedge import KalmanHedge
from strategy.meta_signal import MetaSignalModel, MetaSignalConfig
from utils.pair_id import make_pair_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-cached import of featurize so sys.path manipulation runs only once
# ---------------------------------------------------------------------------
_featurize_fn = None

def _get_featurize():
    global _featurize_fn
    if _featurize_fn is None:
        import sys, os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
        from features.featurize import compute_standard_features
        _featurize_fn = compute_standard_features
    return _featurize_fn

# Default: per-regime target-notional multipliers
DEFAULT_REGIME_SIZE: dict[int, float] = {
    0: 1.0,    # Bull / Low-Vol  — full size
    1: 0.75,   # Neutral         — reduce slightly
    2: 0.4,    # Bear / High-Vol — half size
    3: 0.0,    # Crisis          — no new entries
}

# Regime-adaptive entry z-score thresholds (spec §3.4)
# Low-vol: tighter threshold (mean-reversion is strong, more opportunities)
# High-vol: wider threshold (avoid false entries; spread can stay wide longer)
DEFAULT_REGIME_ENTRY_Z: dict[int, float] = {
    0: 1.5,   # Bull / Low-Vol  — tighter entry
    1: 2.0,   # Neutral         — baseline
    2: 2.5,   # Bear / High-Vol — wider entry (more conservative)
    3: 4.0,   # Crisis          — effectively block new entries
}

# Regime-adaptive exit z-score thresholds (spec §3.4)
# Low-vol: exit close to mean (let trade fully revert)
# High-vol: exit at partial reversion (take profits early, reduce exposure in turbulent markets)
DEFAULT_REGIME_EXIT_Z: dict[int, float] = {
    0: 0.3,   # Bull / Low-Vol  — exit close to mean (full reversion)
    1: 0.5,   # Neutral         — baseline
    2: 0.8,   # Bear / High-Vol — exit at partial reversion (lock in gains quickly)
    3: 1.0,   # Crisis          — exit at any meaningful reversion signal
}


class PairConfig:
    """Data for a single registered pair."""
    def __init__(self, ticker1: str, ticker2: str, hedge_ratio: float,
                 entry_z: float, exit_z: float, stop_z: float):
        self.ticker1      = ticker1
        self.ticker2      = ticker2
        self.hedge_ratio_static  = hedge_ratio  # initial static hedge
        self.hedge_ratio  = hedge_ratio        # current (dynamic) hedge ratio
        self.entry_z      = entry_z
        self.exit_z       = exit_z
        self.stop_z       = stop_z
        self.pair_id      = make_pair_id(ticker1, ticker2)
        self.position: int = 0   # -1 / 0 / +1
        # Kalman filter for dynamic hedge ratio estimation
        self.kalman_hedge = KalmanHedge(initial_hedge=hedge_ratio, process_variance=0.0001)


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
    pair_reselector : optional PairReSelector
        If supplied, periodically re-selects pairs on trailing data.
    all_tickers : list[str], optional
        Full universe of tickers for re-selection price tracking.
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
        regime_entry_z_map: Optional[dict] = None,
        regime_exit_z_map: Optional[dict] = None,
        regime_position_scale_map: Optional[dict] = None,
        warmup_bars: int = 60,
        strategy_id: str = "pairs",
        pair_reselector=None,
        all_tickers: Optional[List[str]] = None,
    ):
        self.zscore_window  = zscore_window
        self.warmup_bars    = warmup_bars
        self.strategy_id    = strategy_id
        self._entry_z       = entry_z
        self._exit_z        = exit_z
        self._stop_z        = stop_z

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

        # Collect all unique tickers (pair tickers + full universe for reselection)
        all_tickers_set: set[str] = set(all_tickers or [])
        for pc in self._pairs:
            all_tickers_set.update([pc.ticker1, pc.ticker2])
        self._tickers = list(all_tickers_set)

        # Price history buffers (enough for z-score + regime window + reselection)
        buf_size = max(zscore_window * 3, 504, 252)
        self._price_buf: Dict[str, deque] = {
            t: deque(maxlen=buf_size) for t in self._tickers
        }
        # Also track dates for building a DataFrame for re-selection
        self._date_buf: deque = deque(maxlen=buf_size)

        # Regime
        self._regime_detector  = regime_detector
        self._regime_ticker    = regime_ticker or (self._tickers[0] if self._tickers else "SPY")
        self._regime_size_map  = regime_size_map or DEFAULT_REGIME_SIZE
        self._regime_entry_z_map = regime_entry_z_map or DEFAULT_REGIME_ENTRY_Z
        self._regime_exit_z_map  = regime_exit_z_map  or DEFAULT_REGIME_EXIT_Z
        self._regime_buf: deque = deque(maxlen=max(zscore_window * 3, 252))
        self._current_regime: int = 1   # start neutral

        # Regime history for analytics (spec §4.4)
        self._regime_history: list = []  # list of (date, regime_label)

        # Pair re-selection
        self._pair_reselector = pair_reselector
        self._reselection_signals: List[SignalEvent] = []

        # Meta-signal model for regime-dependent signal combination and thresholds
        meta_cfg = MetaSignalConfig(
            use_learned_weights=False,
            regime_entry_thresholds=regime_entry_z_map or DEFAULT_REGIME_ENTRY_Z,
            regime_exit_thresholds=regime_exit_z_map or DEFAULT_REGIME_EXIT_Z,
            regime_position_scale=regime_position_scale_map or {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.1},
        )
        self._meta_signal = MetaSignalModel(config=meta_cfg)

        self._bar_count = 0
        # Regime update throttle — regime labels change slowly, no need to
        # re-predict on every single bar.
        self._regime_update_every: int = 5
        self._regime_last_update: int = 0

    # -----------------------------------------------------------------------
    # Called once per bar by the BacktestEngine
    # -----------------------------------------------------------------------

    def on_market_event(
        self,
        event: MarketEvent,
        portfolio: Portfolio,
    ) -> List[SignalEvent]:

        # 1. Update price buffers — single pass over incoming prices.
        # Tickers already in self._tickers but absent from this bar's prices
        # simply don't get a new value appended, which is the same behaviour
        # as the previous two-loop approach.
        _buf_size = max(self.zscore_window * 3, 504, 252)
        for t, p in event.prices.items():
            if t not in self._price_buf:
                self._price_buf[t] = deque(maxlen=_buf_size)
                if t not in self._tickers:
                    self._tickers.append(t)
            if p is not None and np.isfinite(p):
                self._price_buf[t].append(p)

        self._date_buf.append(event.date)
        self._bar_count += 1
        if self._bar_count < self.warmup_bars:
            return []

        # 2. Update regime if detector is attached
        if self._regime_detector is not None:
            self._update_regime(event)

        # Record regime label for analytics (every bar after warmup)
        self._regime_history.append((event.date, self._current_regime))

        # 3. Periodic pair re-selection
        signals = []
        if self._pair_reselector is not None:
            resel_signals = self._try_reselect(event, portfolio)
            signals.extend(resel_signals)

        # 4. Generate signals per pair
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
        
        # Update Kalman hedge ratio with latest prices
        p1_current = s1[-1]
        p2_current = s2[-1]
        pc.hedge_ratio, _ = pc.kalman_hedge.update(p1_current, p2_current)
        
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

        # Regime-adaptive thresholds (spec §3.4) via meta-signal model
        entry_z_eff = self._meta_signal.get_entry_threshold(self._current_regime)
        exit_z_eff  = self._meta_signal.get_exit_threshold(self._current_regime)

        # Exit / stop
        if cur != 0:
            stop_loss_eff = self._meta_signal.get_stop_loss_threshold(self._current_regime)
            if abs(z) < exit_z_eff or abs(z) > stop_loss_eff:
                new_dir = "flat"

        if new_dir is None:
            # Entry — use regime-adaptive entry threshold
            if cur == 0:
                regime_scale = self._meta_signal.get_position_scale(self._current_regime)
                if regime_scale <= 0.0:
                    return None      # regime blocks new entries

                if z < -entry_z_eff:
                    new_dir = "long_spread"
                elif z > entry_z_eff:
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

            regime_scale = self._meta_signal.get_position_scale(self._current_regime) if new_dir != "flat" else 1.0
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
        """Predict regime on the latest price window.

        Throttled to every ``_regime_update_every`` bars because regime labels
        change slowly.  Using only the most-recent 90 prices (enough for
        rv_20/mom_20) and windows=[20] keeps DataFrame construction cheap.
        """
        if self._bar_count - self._regime_last_update < self._regime_update_every:
            return
        self._regime_last_update = self._bar_count

        rt  = self._regime_ticker
        buf = self._price_buf.get(rt)
        if buf is None or len(buf) < self.zscore_window:
            return

        try:
            compute_standard_features = _get_featurize()
            # 90 prices is enough for rv_20 + mom_20 + decent HMM context
            prices = list(buf)[-90:]
            dates  = pd.date_range(end=event.date, periods=len(prices), freq="B")
            mini_df = pd.DataFrame({"Date": dates, "close": prices})
            feat = compute_standard_features(mini_df, windows=[20])

            if "rv_20" not in feat.columns or feat["rv_20"].dropna().empty:
                return

            feat = feat.set_index("Date") if "Date" in feat.columns else feat
            labels = self._regime_detector.predict(feat)
            if len(labels) > 0:
                self._current_regime = int(labels.iloc[-1])
        except Exception as e:
            logger.debug("Regime update skipped: %s", e)

    # -----------------------------------------------------------------------
    # Periodic Pair Re-Selection
    # -----------------------------------------------------------------------

    def _try_reselect(self, event: MarketEvent, portfolio: Portfolio) -> List[SignalEvent]:
        """Check if it's time to re-select pairs, and do so.

        Returns flat signals for any removed pairs.
        """
        # Prefer adaptive re-selection when regime history is available
        recent_regimes = list(r for _, r in self._regime_history[-40:]) if self._regime_history else None
        try:
            should = self._pair_reselector.should_reselect_adaptive(
                self._bar_count,
                current_regime=self._current_regime,
                recent_regimes=recent_regimes,
            )
        except AttributeError:
            # Fallback for older PairReSelector implementations
            should = self._pair_reselector.should_reselect(self._bar_count)

        if not should:
            return []

        # Build a wide price DataFrame from buffers
        price_df = self._build_price_history_df()
        if price_df.empty or price_df.shape[1] < 2:
            return []

        current_pair_ids = {pc.pair_id for pc in self._pairs}

        new_pairs_df, added, removed = self._pair_reselector.reselect(
            bar_count=self._bar_count,
            price_history=price_df,
            current_pair_ids=current_pair_ids,
        )

        if new_pairs_df.empty and not removed:
            return []

        signals: List[SignalEvent] = []

        # O(1) lookup dict — avoids nested O(n*m) loop over pairs
        pairs_by_id: Dict[str, PairConfig] = {pc.pair_id: pc for pc in self._pairs}

        # 1. Close removed pairs
        for pid in removed:
            pc = pairs_by_id.get(pid)
            if pc is not None and pc.position != 0:
                signals.append(SignalEvent(
                    date=event.date,
                    ticker1=pc.ticker1,
                    ticker2=pc.ticker2,
                    direction="flat",
                    strength=1.0,
                    spread_zscore=0.0,
                    hedge_ratio=pc.hedge_ratio,
                    strategy_id=self.strategy_id,
                ))
                pc.position = 0

        # 2. Remove old pair configs and rebuild lookup
        self._pairs = [pc for pc in self._pairs if pc.pair_id not in removed]
        pairs_by_id = {pc.pair_id: pc for pc in self._pairs}

        # 3. Add new pairs / update hedge ratios — itertuples avoids row-copy overhead
        _buf_size = max(self.zscore_window * 3, 504, 252)
        for row in new_pairs_df.itertuples(index=False):
            pid = f"{row.ticker1}/{row.ticker2}"
            if pid in pairs_by_id:
                # Update hedge ratio for existing pair and reset Kalman
                pairs_by_id[pid].hedge_ratio = row.hedge_ratio
                pairs_by_id[pid].hedge_ratio_static = row.hedge_ratio
                pairs_by_id[pid].kalman_hedge.reset(initial_hedge=row.hedge_ratio)
            elif pid in added:
                # Add new pair
                new_pc = PairConfig(
                    ticker1=row.ticker1,
                    ticker2=row.ticker2,
                    hedge_ratio=row.hedge_ratio,
                    entry_z=self._entry_z,
                    exit_z=self._exit_z,
                    stop_z=self._stop_z,
                )
                self._pairs.append(new_pc)
                pairs_by_id[pid] = new_pc
                # Ensure price buffers exist for new tickers
                for t in [row.ticker1, row.ticker2]:
                    if t not in self._price_buf:
                        self._price_buf[t] = deque(maxlen=_buf_size)
                    if t not in self._tickers:
                        self._tickers.append(t)

        logger.info("Post-reselection: %d active pairs", len(self._pairs))
        return signals

    def get_regime_history(self) -> pd.Series:
        """Return regime label per trading bar as a Series (for analytics/plotting)."""
        if not self._regime_history:
            return pd.Series(dtype=int)
        dates, labels = zip(*self._regime_history)
        return pd.Series(list(labels), index=list(dates), name="regime", dtype=int)

    def _build_price_history_df(self) -> pd.DataFrame:
        """Build a wide price DataFrame from the internal price buffers."""
        if not self._date_buf:
            return pd.DataFrame()

        dates = list(self._date_buf)
        n = len(dates)
        data = {}
        for t, buf in self._price_buf.items():
            prices = list(buf)
            if len(prices) == n:
                data[t] = prices
            elif len(prices) < n:
                # Pad front with NaN
                data[t] = [float("nan")] * (n - len(prices)) + prices
            else:
                # Trim to match dates
                data[t] = prices[-n:]

        if not data:
            return pd.DataFrame()

        return pd.DataFrame(data, index=dates)
