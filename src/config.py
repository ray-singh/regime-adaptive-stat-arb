"""Centralized configuration for the Regime-Adaptive Stat-Arb platform.

Usage:
    from config import PlatformConfig
    cfg = PlatformConfig()                         # all defaults
    cfg = PlatformConfig.from_yaml("config.yaml")  # from file
    cfg = PlatformConfig.from_env()                # from environment variables
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List
# Use the predefined universe when available
try:
    from data.universe import TOP_200_LIQUID_US_EQUITIES
except Exception:
    TOP_200_LIQUID_US_EQUITIES = None


@dataclass
class DataConfig:
    """Data fetching settings."""
    tickers: List[str] = field(default_factory=(lambda: TOP_200_LIQUID_US_EQUITIES if TOP_200_LIQUID_US_EQUITIES is not None else [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
        "TSLA", "JPM", "V", "UNH", "ADBE", "AMD", "AVGO", "CRM", "ORCL",
    ]))
    period: str = "10y"
    interval: str = "1d"
    cache_dir: str = "data/cache"


@dataclass
class RegimeConfig:
    """Regime detection settings."""
    n_states: int = 3
    regime_ticker: str = "VOO"
    # Walk-forward training (guide §3) — prevents look-ahead bias
    use_walkforward: bool = True
    walkforward_min_train_years: int = 5   # minimum bars before first prediction window
    walkforward_retrain_years: int = 1     # refit every N years with expanding window
    # Multivariate macro HMM (guide §6) — leave empty for univariate mode
    # Example: ["^VIX", "GLD", "TLT", "USO"]
    macro_tickers: List[str] = field(default_factory=list)


@dataclass
class PairsConfig:
    """Pairs trading and selection settings."""
    pvalue_threshold: float = 0.05
    min_half_life: int = 5
    max_half_life: int = 126
    max_pairs: int = 10
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    warmup_bars: int = 60
    # Regime-adaptive z-score thresholds (spec §3.4)
    # Keys: regime label (0=low-vol, 1=neutral, 2=high-vol, 3=crisis)
    # Low-vol: tighter entry (richer mean-reversion), High-vol: wider entry (fewer false signals)
    regime_entry_z: dict = field(default_factory=lambda: {0: 1.5, 1: 2.0, 2: 2.5, 3: 4.0})
    # Low-vol: exit close to mean, High-vol: exit earlier to lock in partial gains
    regime_exit_z: dict = field(default_factory=lambda: {0: 0.3, 1: 0.5, 2: 0.8, 3: 1.0})


@dataclass
class ReselectionConfig:
    """Periodic pair re-selection settings."""
    enabled: bool = True
    interval_days: int = 63
    lookback_days: int = 504


@dataclass
class ExecutionConfigSpec:
    """Execution / broker settings."""
    slippage_bps: float = 5.0
    spread_bps: float = 3.0
    commission_pct: float = 0.001
    min_commission: float = 1.0


@dataclass
class RiskConfigSpec:
    """Risk management settings."""
    max_gross_leverage: float = 4.0
    max_net_leverage: float = 2.0
    max_pair_notional_pct: float = 0.20
    max_ticker_notional_pct: float = 0.25
    max_open_pairs: int = 10
    drawdown_halt_pct: float = -0.30
    drawdown_reduce_pct: float = -0.15
    drawdown_scale_factor: float = 0.50
    # Regime-aware risk maps (optional)
    regime_leverage_caps: dict = field(default_factory=lambda: {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0})
    regime_max_open_pairs: dict = field(default_factory=lambda: {0: 10, 1: 8, 2: 5, 3: 2})
    regime_pair_notional_pct: dict = field(default_factory=lambda: {0: 0.2, 1: 0.15, 2: 0.10, 3: 0.05})
    regime_ticker_notional_pct: dict = field(default_factory=lambda: {0: 0.25, 1: 0.20, 2: 0.15, 3: 0.08})


@dataclass
class BacktestConfig:
    """Backtesting settings."""
    initial_capital: float = 1_000_000.0
    train_pct: float = 0.50
    target_notional_pct: float = 0.10
    verbose: bool = True


@dataclass
class PlatformConfig:
    """Top-level configuration aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    reselection: ReselectionConfig = field(default_factory=ReselectionConfig)
    execution: ExecutionConfigSpec = field(default_factory=ExecutionConfigSpec)
    risk: RiskConfigSpec = field(default_factory=RiskConfigSpec)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    plots_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "plots",
    ))
    log_level: str = "INFO"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "PlatformConfig":
        """Load config from a YAML file (requires PyYAML)."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML config files")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        for section_name in ["data", "regime", "pairs", "reselection",
                             "execution", "risk", "backtest"]:
            if section_name in raw:
                section = getattr(cfg, section_name)
                for k, v in raw[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)

        if "plots_dir" in raw:
            cfg.plots_dir = raw["plots_dir"]
        if "log_level" in raw:
            cfg.log_level = raw["log_level"]

        return cfg

    @classmethod
    def from_env(cls) -> "PlatformConfig":
        """Override defaults from environment variables.

        Env vars follow pattern: STATARB_<SECTION>_<KEY> (uppercase).
        E.g. STATARB_BACKTEST_INITIAL_CAPITAL=2000000
        """
        cfg = cls()

        prefix = "STATARB_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue
            section_name, field_name = parts
            section = getattr(cfg, section_name, None)
            if section is None:
                continue
            if not hasattr(section, field_name):
                continue

            # Coerce type
            current = getattr(section, field_name)
            try:
                if isinstance(current, bool):
                    setattr(section, field_name, value.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(section, field_name, int(value))
                elif isinstance(current, float):
                    setattr(section, field_name, float(value))
                else:
                    setattr(section, field_name, value)
            except (ValueError, TypeError):
                pass

        return cfg


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure structured logging for the platform.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str, optional
        If provided, also log to this file.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "yfinance", "matplotlib", "hmmlearn"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
