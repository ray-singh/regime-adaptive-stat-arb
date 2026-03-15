from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure src/ is importable when running from dashboard/backend
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import logging

from config import PlatformConfig, setup_logging
from backtest.job_queue import BacktestJobQueue, JobStatus

logger = logging.getLogger(__name__)


@dataclass
class BacktestStore:
    """In-memory state for latest dashboard backtest."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    last_error: Optional[str] = None
    last_result: Optional[dict[str, Any]] = None
    cache: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryStore:
    """In-memory state for the pair-discovery pipeline."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    last_error: Optional[str] = None
    result: Optional[dict[str, Any]] = None  # keys: pairs_df, pair_summary, ranked, regime_series, price_matrix


store = BacktestStore()
discovery_store = DiscoveryStore()


@lru_cache(maxsize=8)
def _get_cached_price_matrix(tickers_key: str, start_date: str, end_date: Optional[str]) -> Any:
    from data.yfinance_client import YFinanceClient

    tickers = [t for t in tickers_key.split(",") if t]
    client = YFinanceClient(cache_dir=os.path.join(ROOT_DIR, "data", "cache"))
    return client.get_price_matrix(tickers, start_date=start_date, end_date=end_date)


@lru_cache(maxsize=8)
def _get_cached_market_features(
    tickers_key: str,
    start_date: str,
    end_date: Optional[str],
    last_date: str,
) -> Any:
    from features.featurize import compute_market_features

    price_matrix = _get_cached_price_matrix(tickers_key, start_date, end_date)
    return compute_market_features(price_matrix)


def _get_hmm_cache_key(tickers: list, start_date: str, end_date: Optional[str], n_states: int) -> str:
    """Generate a cache key for HMM regime detection results."""
    key_data = {
        "tickers": sorted(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "n_states": n_states,
        "feature_cols": ["avg_rv", "ret_dispersion", "index_momentum"],
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_pairs_cache_key(tickers: list, regime_series_hash: str, min_regime_bars: int) -> str:
    """Generate a cache key for pair discovery results."""
    key_data = {
        "tickers": sorted(tickers),
        "regime_series_hash": regime_series_hash,
        "min_regime_bars": min_regime_bars,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_ranking_cache_key(tickers: list, regime_series_hash: str, pair_summary_hash: str) -> str:
    """Generate a cache key for pair ranking results."""
    key_data = {
        "tickers": sorted(tickers),
        "regime_series_hash": regime_series_hash,
        "pair_summary_hash": pair_summary_hash,
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()


def _compute_data_hash(data: Any) -> str:
    """Compute MD5 hash of serializable data (for cache invalidation)."""
    try:
        import pickle
        return hashlib.md5(pickle.dumps(data)).hexdigest()
    except Exception:
        return ""


def _load_cache_with_ttl(cache_key: str, cache_dir: Path, prefix: str, ttl_seconds: int = 604800) -> dict | None:
    """Load cached data if it exists and is not stale (TTL in seconds; default 7 days)."""
    cache_file = cache_dir / f"{prefix}_{cache_key}.pkl"
    if cache_file.exists():
        try:
            import pickle
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            timestamp = cached.get("timestamp", 0)
            age_seconds = time.time() - timestamp
            if age_seconds < ttl_seconds:
                logger.info(f"Cache hit for {prefix}_{cache_key} (age: {age_seconds:.1f}s, TTL: {ttl_seconds}s)")
                return cached.get("data")
            else:
                logger.info(f"Cache expired for {prefix}_{cache_key} (age: {age_seconds:.1f}s, TTL: {ttl_seconds}s)")
                cache_file.unlink(missing_ok=True)  # Remove stale cache
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
    return None


def _save_cache_with_ttl(cache_key: str, cache_dir: Path, prefix: str, data: dict) -> None:
    """Save data to cache with timestamp for TTL management."""
    cache_file = cache_dir / f"{prefix}_{cache_key}.pkl"
    try:
        import pickle
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_obj = {"timestamp": time.time(), "data": data}
        with open(cache_file, "wb") as f:
            pickle.dump(cached_obj, f)
        logger.info(f"Saved cache to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")


def _load_hmm_cache(cache_key: str, cache_dir: Path) -> dict | None:
    """Load cached HMM regime_series and market_features if available (TTL: 7 days)."""
    return _load_cache_with_ttl(cache_key, cache_dir, "hmm", ttl_seconds=604800)


def _save_hmm_cache(cache_key: str, cache_dir: Path, data: dict) -> None:
    """Save HMM regime_series and market_features to cache with timestamp."""
    _save_cache_with_ttl(cache_key, cache_dir, "hmm", data)


def _load_pairs_cache(cache_key: str, cache_dir: Path) -> dict | None:
    """Load cached pair discovery results if available (TTL: 7 days)."""
    return _load_cache_with_ttl(cache_key, cache_dir, "pairs", ttl_seconds=604800)


def _save_pairs_cache(cache_key: str, cache_dir: Path, data: dict) -> None:
    """Save pair discovery results to cache with timestamp."""
    _save_cache_with_ttl(cache_key, cache_dir, "pairs", data)


def _load_ranking_cache(cache_key: str, cache_dir: Path) -> dict | None:
    """Load cached pair ranking results if available (TTL: 7 days)."""
    return _load_cache_with_ttl(cache_key, cache_dir, "ranking", ttl_seconds=604800)


def _save_ranking_cache(cache_key: str, cache_dir: Path, data: dict) -> None:
    """Save pair ranking results to cache with timestamp."""
    _save_cache_with_ttl(cache_key, cache_dir, "ranking", data)


def _run_discovery_pipeline(payload: dict) -> dict:
    """Run the full pair-discovery pipeline and return all intermediate results."""
    import pandas as pd
    from regime.hmm_detector import HMMRegimeDetector
    from pair_discovery import PairDiscoveryEngine
    from relationship_analysis import RelationshipAnalyzer
    from pair_ranking import PairRankingEngine

    cfg = PlatformConfig()
    tickers: list = payload.get("tickers") or getattr(cfg.data, "tickers", []) or []
    start_date: str = payload.get("startDate") or getattr(cfg.data, "start_date", "2018-01-01") or "2018-01-01"
    end_date: Optional[str] = payload.get("endDate") or getattr(cfg.data, "end_date", None)
    n_states: int = int(payload.get("nStates") or getattr(cfg.regime, "n_states", 4) or 4)
    min_train_years: int = int(payload.get("minTrainYears") or 2)
    min_regime_bars: int = int(payload.get("minRegimeBars") or 30)

    if not tickers:
        raise ValueError("No tickers configured. Pass 'tickers' in the request body or set DataConfig.tickers.")

    cache_dir = Path(ROOT_DIR) / "data" / "cache"

    # 1. Fetch price matrix (in-process LRU + disk cache)
    tickers_key = ",".join(sorted(str(t) for t in tickers))
    price_matrix = _get_cached_price_matrix(tickers_key, start_date, end_date).copy()
    if price_matrix.empty:
        raise ValueError("Price matrix is empty — check tickers and date range.")

    last_date = str(price_matrix.index[-1].date())
    cache_key = _get_hmm_cache_key(tickers, start_date, last_date, n_states)

    # 2–3. Check cache for market_features and regime_series (skip expensive HMM fitting if available)
    cached = _load_hmm_cache(cache_key, cache_dir)
    if cached is not None:
        logger.info(f"Using cached HMM results for key {cache_key}")
        market_features = cached["market_features"]
        regime_series = cached["regime_series"]
    else:
        logger.info(f"Computing HMM regime detection for key {cache_key}")
        # Compute cross-asset market features (in-process cache)
        market_features = _get_cached_market_features(
            tickers_key=tickers_key,
            start_date=start_date,
            end_date=end_date,
            last_date=last_date,
        ).copy()

        # Detect regimes using market features
        hmm_feature_cols = ["avg_rv", "ret_dispersion", "index_momentum"]
        detector = HMMRegimeDetector(n_states=n_states, feature_cols=hmm_feature_cols)
        feature_df = market_features[hmm_feature_cols].dropna()
        regime_series: pd.Series = detector.fit_predict_walkforward(
            feature_df, min_train_years=min_train_years
        )
        regime_series = regime_series.dropna().astype(int)
        
        # Save to cache
        _save_hmm_cache(cache_key, cache_dir, {"market_features": market_features, "regime_series": regime_series})

    # 4. Discover pairs per regime (with caching)
    regime_series_hash = _compute_data_hash(regime_series)
    pairs_cache_key = _get_pairs_cache_key(tickers, regime_series_hash, min_regime_bars)
    
    cached_pairs = _load_pairs_cache(pairs_cache_key, cache_dir)
    if cached_pairs is not None:
        logger.info(f"Using cached pairs discovery for key {pairs_cache_key}")
        pairs_df = cached_pairs["pairs_df"]
        pair_regime_stats = cached_pairs["pair_regime_stats"]
        pair_summary = cached_pairs["pair_summary"]
    else:
        logger.info(f"Computing pair discovery for key {pairs_cache_key}")
        engine = PairDiscoveryEngine(min_regime_bars=min_regime_bars)
        pairs_df = engine.discover(price_matrix, regime_series, tickers=tickers)

        # 5. Analyse relationships
        analyzer = RelationshipAnalyzer(price_matrix=price_matrix)
        pair_regime_stats, pair_summary = analyzer.analyse(pairs_df, regime_series)
        
        # Save pairs to cache
        _save_pairs_cache(pairs_cache_key, cache_dir, {
            "pairs_df": pairs_df,
            "pair_regime_stats": pair_regime_stats,
            "pair_summary": pair_summary,
        })

    # 6. Rank pairs (with caching)
    pair_summary_hash = _compute_data_hash(pair_summary)
    ranking_cache_key = _get_ranking_cache_key(tickers, regime_series_hash, pair_summary_hash)
    
    cached_ranking = _load_ranking_cache(ranking_cache_key, cache_dir)
    if cached_ranking is not None:
        logger.info(f"Using cached pair ranking for key {ranking_cache_key}")
        ranked = cached_ranking["ranked"]
    else:
        logger.info(f"Computing pair ranking for key {ranking_cache_key}")
        ranker = PairRankingEngine()
        ranked = ranker.rank(pair_summary) if not pair_summary.empty else pair_summary
        
        # Save ranking to cache
        _save_ranking_cache(ranking_cache_key, cache_dir, {"ranked": ranked})

    return {
        "pairs_df": pairs_df,
        "pair_regime_stats": pair_regime_stats,
        "pair_summary": pair_summary,
        "ranked": ranked,
        "regime_series": regime_series,
        "price_matrix": price_matrix,
        "market_features": market_features,
    }


def _run_backtest_from_payload(payload: dict) -> dict:
    """Thin adapter so BacktestJobQueue can call run_backtest."""
    # Import here to avoid importing heavy backtest dependencies at module
    # import time (which can fail inside minimal container images).
    from backtest.run_backtest import run_backtest

    cfg, use_risk = _build_config_from_payload(payload)
    return run_backtest(cfg, use_risk=use_risk)


# Global job queue — max 2 concurrent long backtests
job_queue = BacktestJobQueue(runner=_run_backtest_from_payload, max_workers=2)


def _build_config_from_payload(payload: dict[str, Any]) -> tuple[PlatformConfig, bool]:
    config_path = payload.get("configPath")
    use_risk = bool(payload.get("useRisk", True))

    if config_path:
        config_abs = os.path.abspath(os.path.join(ROOT_DIR, config_path))
        cfg = PlatformConfig.from_yaml(config_abs)
    else:
        cfg = PlatformConfig()

    overrides = payload.get("overrides", {}) or {}

    if "initialCapital" in overrides:
        cfg.backtest.initial_capital = float(overrides["initialCapital"])
    if "trainPct" in overrides:
        cfg.backtest.train_pct = float(overrides["trainPct"])
    if "maxPairs" in overrides:
        cfg.pairs.max_pairs = int(overrides["maxPairs"])
    if "reselectionInterval" in overrides:
        cfg.reselection.interval_days = int(overrides["reselectionInterval"])
    if "reselectionEnabled" in overrides:
        cfg.reselection.enabled = bool(overrides["reselectionEnabled"])
    if "entryZ" in overrides:
        cfg.pairs.entry_z = float(overrides["entryZ"])
    if "exitZ" in overrides:
        cfg.pairs.exit_z = float(overrides["exitZ"])
    if "stopZ" in overrides:
        cfg.pairs.stop_z = float(overrides["stopZ"])
    # Regime-adaptive threshold overrides (spec §3.4)
    if "regimeEntryZ" in overrides:
        raw = overrides["regimeEntryZ"]
        if isinstance(raw, dict):
            cfg.pairs.regime_entry_z = {int(k): float(v) for k, v in raw.items()}
    if "regimeExitZ" in overrides:
        raw = overrides["regimeExitZ"]
        if isinstance(raw, dict):
            cfg.pairs.regime_exit_z = {int(k): float(v) for k, v in raw.items()}
    if "regimePositionScale" in overrides:
        raw = overrides["regimePositionScale"]
        if isinstance(raw, dict):
            try:
                cfg.pairs.regime_position_scale = {int(k): float(v) for k, v in raw.items()}
            except Exception:
                pass
    if "nStates" in overrides:
        cfg.regime.n_states = int(overrides["nStates"])

    # Allow toggling macro tickers from the dashboard: overrides.macroTickers
    if "macroTickers" in overrides:
        mt = overrides.get("macroTickers") or []
        try:
            if isinstance(mt, list):
                cfg.regime.macro_tickers = [str(x) for x in mt]
            elif isinstance(mt, str):
                cfg.regime.macro_tickers = [mt]
        except Exception:
            pass

    # Regime-aware risk maps (from frontend overrides)
    if "regimeLeverageCaps" in overrides:
        raw = overrides.get("regimeLeverageCaps") or {}
        if isinstance(raw, dict):
            try:
                cfg.risk.regime_leverage_caps = {int(k): float(v) for k, v in raw.items()}
            except Exception:
                pass
    if "regimeMaxOpenPairs" in overrides:
        raw = overrides.get("regimeMaxOpenPairs") or {}
        if isinstance(raw, dict):
            try:
                cfg.risk.regime_max_open_pairs = {int(k): int(v) for k, v in raw.items()}
            except Exception:
                pass
    if "regimePairNotionalPct" in overrides:
        raw = overrides.get("regimePairNotionalPct") or {}
        if isinstance(raw, dict):
            try:
                cfg.risk.regime_pair_notional_pct = {int(k): float(v) for k, v in raw.items()}
            except Exception:
                pass
    if "regimeTickerNotionalPct" in overrides:
        raw = overrides.get("regimeTickerNotionalPct") or {}
        if isinstance(raw, dict):
            try:
                cfg.risk.regime_ticker_notional_pct = {int(k): float(v) for k, v in raw.items()}
            except Exception:
                pass

    # Universe override (top-level or inside overrides) -- allows selecting sector or full top200
    universe = payload.get("universe") or overrides.get("universe")
    if universe:
        try:
            from data.universe import get_universe, SECTOR_MAPPING, get_sector_tickers
        except Exception:
            get_universe = None
            get_sector_tickers = None
            SECTOR_MAPPING = {}

        if get_universe is not None:
            u = str(universe).strip()
            try:
                if u.lower() in ("top200", "top-200", "all"):
                    cfg.data.tickers = get_universe("top200")
                elif u.lower() in ("top100", "top-100"):
                    cfg.data.tickers = get_universe("top100")
                else:
                    # Try case-insensitive sector match
                    match = None
                    for k in list(SECTOR_MAPPING.keys()):
                        if k.lower() == u.lower():
                            match = k
                            break
                    if match and get_sector_tickers is not None:
                        cfg.data.tickers = get_sector_tickers(match)
            except Exception:
                # ignore and keep defaults
                pass

    return cfg, use_risk


def _to_jsonable(value: Any) -> Any:
    """Convert pandas/numpy-heavy structures into JSON-safe objects."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None
        pd = None

    if value is None:
        return None

    if pd is not None and isinstance(value, pd.Series):
        return [
            {"date": str(idx), "value": float(v)}
            for idx, v in value.dropna().items()
        ]

    if pd is not None and isinstance(value, pd.DataFrame):
        rows = value.reset_index().to_dict("records")
        return _to_jsonable(rows)

    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]

    if np is not None and isinstance(value, np.generic):
        return value.item()

    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass

    return value


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    setup_logging(level="INFO")

    @app.get("/api/health")
    def health() -> Any:
        with store.lock:
            return jsonify({
                "ok": True,
                "running": store.running,
                "hasResult": store.last_result is not None,
                "lastError": store.last_error,
            })

    @app.post("/api/run-backtest")
    def run_backtest_endpoint() -> Any:
        payload = request.get_json(silent=True) or {}

        with store.lock:
            if store.running:
                return jsonify({"ok": False, "error": "Backtest already running"}), 409
            store.running = True
            store.last_error = None
            store.cache.clear()

        try:
            from backtest.run_backtest import run_backtest
            cfg, use_risk = _build_config_from_payload(payload)
            result = run_backtest(cfg, use_risk=use_risk)
            with store.lock:
                store.last_result = result
                store.cache.clear()
            return jsonify({
                "ok": True,
                "summary": _to_jsonable(result.get("stats", {})),
                "risk": _to_jsonable(result.get("risk_summary", {})),
                "pairReselections": result.get("pair_reselection_count", 0),
            })
        except Exception as exc:
            with store.lock:
                store.last_error = str(exc)
            return jsonify({"ok": False, "error": str(exc)}), 500
        finally:
            with store.lock:
                store.running = False

    @app.get("/api/summary")
    def summary() -> Any:
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            result = store.last_result

        return jsonify({
            "ok": True,
            "stats": _to_jsonable(result.get("stats", {})),
            "risk": _to_jsonable(result.get("risk_summary", {})),
            "pairReselections": result.get("pair_reselection_count", 0),
            "selectedPairs": _to_jsonable(result.get("selected_pairs", [])),
        })

    @app.get("/api/equity")
    def equity() -> Any:
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            equity_series = store.last_result.get("equity_curve")
        return jsonify({"ok": True, "equity": _to_jsonable(equity_series)})

    @app.get("/api/trades")
    def trades() -> Any:
        limit = int(request.args.get("limit", 200))
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            trades_df = store.last_result.get("trades")

        rows = _to_jsonable(trades_df) if trades_df is not None else []
        rows = rows[-limit:] if isinstance(rows, list) else []
        return jsonify({"ok": True, "trades": rows})

    @app.get("/api/pairs")
    def pairs() -> Any:
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            pairs_data = store.last_result.get("selected_pairs", [])
        return jsonify({"ok": True, "pairs": _to_jsonable(pairs_data)})

    @app.get("/api/trades-pnl")
    def trades_with_pnl() -> Any:
        """Return trades enriched with per-leg cashflow and closed round-trip PnL grouping."""
        limit = int(request.args.get("limit", 500))
        closed_limit = int(request.args.get("closedLimit", 500))

        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            trades_df = store.last_result.get("trades")
            cached = store.cache.get("trades_pnl_full")

        if cached is not None:
            trades_rows = cached.get("trades", [])
            closed_rows = cached.get("closedTrades", [])
            return jsonify({
                "ok": True,
                "trades": trades_rows[-limit:] if limit > 0 else trades_rows,
                "closedTrades": closed_rows[-closed_limit:] if closed_limit > 0 else closed_rows,
            })

        if trades_df is None:
            return jsonify({"ok": True, "trades": [], "closedTrades": []})

        try:
            import pandas as pd
            df = trades_df.copy()
            # if date is in the index, reset it to a column
            if hasattr(df, "index") and (getattr(df.index, "name", None) == "date" or "date" in (df.index.names or [])):
                df = df.reset_index()
            # Ensure columns exist
            for c in ["date", "pair_id", "quantity", "fill_price", "commission"]:
                if c not in df.columns:
                    df[c] = None

            if "date" in df.columns:
                df = df.sort_values(by="date").reset_index(drop=True)
            else:
                df = df.reset_index().sort_values(by="index").reset_index(drop=True)

            enriched = []
            closed = []
            round_id = 0

            # group by pair_id and detect round-trip closures when running_qty returns to zero
            for pair, grp in df.groupby("pair_id", sort=False):
                running_qty = 0.0
                acc_cash = 0.0
                start_date = None
                legs = []
                for row in grp.itertuples(index=False):
                    qty = float(row.quantity or 0)
                    price = float(row.fill_price or 0)
                    commission = float(row.commission or 0)
                    cash_flow = -qty * price - commission
                    if start_date is None:
                        start_date = row.date
                    running_qty += qty
                    acc_cash += cash_flow
                    legs.append({
                        "date": str(row.date),
                        "ticker": getattr(row, "ticker", None),
                        "quantity": qty,
                        "fill_price": price,
                        "commission": commission,
                        "cash_flow": float(cash_flow),
                        "pair_id": pair,
                    })

                    enriched.append({**legs[-1], "roundtrip_id": round_id if running_qty == 0 else None})

                    if abs(running_qty) < 1e-9:
                        # Closed round-trip
                        end_date = row.get("date")
                        realized_pnl = float(acc_cash)
                        closed.append({
                            "roundtrip_id": round_id,
                            "pair_id": pair,
                            "start_date": str(start_date),
                            "end_date": str(end_date),
                            "realized_pnl": realized_pnl,
                            "legs": len(legs),
                        })
                        round_id += 1
                        # reset
                        running_qty = 0.0
                        acc_cash = 0.0
                        start_date = None
                        legs = []

            payload = {"trades": _to_jsonable(enriched), "closedTrades": _to_jsonable(closed)}
            with store.lock:
                store.cache["trades_pnl_full"] = payload

            trades_rows = payload["trades"]
            closed_rows = payload["closedTrades"]
            return jsonify({
                "ok": True,
                "trades": trades_rows[-limit:] if limit > 0 else trades_rows,
                "closedTrades": closed_rows[-closed_limit:] if closed_limit > 0 else closed_rows,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/regime")
    def regime() -> Any:
        """Return per-bar regime label series (spec §4.1 — regime timeline)."""
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            regime_series = store.last_result.get("regime_series")
        if regime_series is None:
            return jsonify({"ok": True, "regime": []})
        return jsonify({"ok": True, "regime": _to_jsonable(regime_series)})

    @app.get("/api/regime-performance")
    def regime_performance() -> Any:
        """Return per-regime performance breakdown (spec §4.4)."""
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            perf_df = store.last_result.get("regime_performance")
        if perf_df is None:
            return jsonify({"ok": True, "regimePerformance": []})
        try:
            import pandas as pd
            rows = perf_df.reset_index().to_dict("records")
            return jsonify({"ok": True, "regimePerformance": _to_jsonable(rows)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/drawdown")
    def drawdown() -> Any:
        """Return pre-computed drawdown series (pct from peak)."""
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            equity_series = store.last_result.get("equity_curve")
        if equity_series is None:
            return jsonify({"ok": True, "drawdown": []})
        try:
            import pandas as pd
            eq = equity_series
            roll_max = eq.cummax()
            dd = (eq - roll_max) / roll_max * 100
            result = [{"date": str(idx), "value": float(v)} for idx, v in dd.items()]
            return jsonify({"ok": True, "drawdown": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/rolling-sharpe")
    def rolling_sharpe() -> Any:
        """Return rolling 60-day Sharpe ratio series."""
        window = int(request.args.get("window", 60))
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            equity_series = store.last_result.get("equity_curve")
        if equity_series is None:
            return jsonify({"ok": True, "rollingSharpe": []})
        try:
            import math
            rets = equity_series.pct_change().dropna()
            roll_mean = rets.rolling(window).mean()
            roll_std = rets.rolling(window).std()
            sharpe = roll_mean / roll_std.replace(0, float("nan")) * math.sqrt(252)
            result = [
                {"date": str(idx), "value": float(v)}
                for idx, v in sharpe.dropna().items()
            ]
            return jsonify({"ok": True, "rollingSharpe": result})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/hmm")
    def hmm_info() -> Any:
        """Return HMM debug info collected during the last backtest run."""
        with store.lock:
            if store.last_result is None:
                return jsonify({"ok": False, "error": "No backtest run yet"}), 404
            info = store.last_result.get("hmm_info")
        if info is None:
            return jsonify({"ok": True, "hmm_info": {}})
        return jsonify({"ok": True, "hmm_info": _to_jsonable(info)})

    # ── Job queue endpoints ────────────────────────────────────────────────

    @app.post("/api/jobs/submit")
    def jobs_submit() -> Any:
        """Submit a backtest job asynchronously.  Returns job_id immediately.

        Accepts the same JSON payload as /api/run-backtest.
        Poll /api/jobs/<job_id> for status and results.
        """
        payload = request.get_json(silent=True) or {}
        try:
            job_id = job_queue.submit(payload)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, "job_id": job_id, "status": "pending"}), 202

    @app.get("/api/jobs")
    def jobs_list() -> Any:
        """List all backtest jobs with their status (no result blobs)."""
        limit = int(request.args.get("limit", 50))
        return jsonify({"ok": True, "jobs": job_queue.list_jobs(limit=limit)})

    @app.get("/api/jobs/<job_id>")
    def jobs_get(job_id: str) -> Any:
        """Return full status + result for a single job.

        Result blob is included only when status == 'complete'.
        """
        job = job_queue.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found"}), 404

        body: dict[str, Any] = {"ok": True, **job.to_summary()}

        if job.status == JobStatus.COMPLETE and job.result is not None:
            body["summary"] = _to_jsonable(job.result.get("stats", {}))
            body["risk"]    = _to_jsonable(job.result.get("risk_summary", {}))
            body["pairReselections"] = job.result.get("pair_reselection_count", 0)

        return jsonify(body)

    @app.delete("/api/jobs/<job_id>")
    def jobs_cancel(job_id: str) -> Any:
        """Cancel a pending job.  Returns 409 if the job is already running."""
        job = job_queue.get(job_id)
        if job is None:
            return jsonify({"ok": False, "error": "Job not found"}), 404
        if job.status.value not in ("pending",):
            return jsonify({"ok": False, "error": f"Cannot cancel a {job.status.value} job"}), 409
        ok = job_queue.cancel(job_id)
        return jsonify({"ok": ok})

    # ── Pair-discovery endpoints (spec §4–6) ────────────────────────────────

    @app.post("/api/discover")
    def start_discovery() -> Any:
        """Start the pair-discovery pipeline in a background thread.

        Accepts JSON body with optional fields:
          - tickers : list[str]   (overrides config default)
          - startDate : str       (YYYY-MM-DD)
          - endDate   : str       (YYYY-MM-DD, optional)
          - nStates   : int       (number of HMM regimes, default 4)
        """
        payload = request.get_json(silent=True) or {}
        with discovery_store.lock:
            if discovery_store.running:
                return jsonify({"ok": False, "error": "Discovery already running"}), 409
            discovery_store.running = True
            discovery_store.last_error = None

        def _run() -> None:
            try:
                result = _run_discovery_pipeline(payload)
                with discovery_store.lock:
                    discovery_store.result = result
            except Exception as exc:
                logger.exception("Discovery pipeline error")
                with discovery_store.lock:
                    discovery_store.last_error = str(exc)
            finally:
                with discovery_store.lock:
                    discovery_store.running = False

        threading.Thread(target=_run, daemon=True).start()
        return jsonify({"ok": True, "status": "running"}), 202

    @app.get("/api/discovery/status")
    def discovery_status() -> Any:
        """Poll the current state of the discovery pipeline."""
        with discovery_store.lock:
            return jsonify({
                "ok": True,
                "running": discovery_store.running,
                "hasResult": discovery_store.result is not None,
                "error": discovery_store.last_error,
            })

    @app.get("/api/pairs/ranked")
    def ranked_pairs() -> Any:
        """Return the ranked pairs table (spec §6)."""
        with discovery_store.lock:
            result = discovery_store.result
        if result is None:
            return jsonify({"ok": False, "error": "No discovery run yet"}), 404
        ranked = result.get("ranked")
        if ranked is None or ranked.empty:
            return jsonify({"ok": True, "pairs": []})
        return jsonify({"ok": True, "pairs": _to_jsonable(ranked.reset_index(drop=True))})

    @app.get("/api/pairs/by-regime")
    def pairs_by_regime() -> Any:
        """Return discovered pairs grouped by regime (spec — Regime Comparison View)."""
        with discovery_store.lock:
            result = discovery_store.result
        if result is None:
            return jsonify({"ok": False, "error": "No discovery run yet"}), 404
        pairs_df = result.get("pairs_df")
        if pairs_df is None or pairs_df.empty:
            return jsonify({"ok": True, "byRegime": {}})
        # Merge ranking metadata (score, stability_label) when available so
        # frontend regime-filtered lists include the same ranking fields as
        # the global `ranked` endpoint.
        ranked = result.get("ranked")
        ranked_map = None
        if ranked is not None and not getattr(ranked, "empty", True):
            try:
                ranked_map = dict((r["pair_id"], {"score": r.get("score"), "stability_label": r.get("stability_label")}) for r in ranked.reset_index(drop=True).to_dict("records"))
            except Exception:
                ranked_map = None

        by_regime: dict[str, Any] = {}
        for regime, grp in pairs_df.groupby("regime"):
            grp_out = grp.sort_values("coint_pvalue").head(20).reset_index(drop=True)
            # If we have ranking info, attach it to each row (non-destructive)
            if ranked_map is not None:
                try:
                    rows = []
                    for _, r in grp_out.iterrows():
                        pid = r.get("pair_id")
                        meta = ranked_map.get(pid) if pid is not None else None
                        row = r.to_dict()
                        if meta:
                            row.update(meta)
                        rows.append(row)
                    by_regime[str(int(regime))] = _to_jsonable(rows)
                    continue
                except Exception:
                    pass
            by_regime[str(int(regime))] = _to_jsonable(grp_out)
        return jsonify({"ok": True, "byRegime": by_regime})

    @app.get("/api/pairs/<path:pair_id>/spread")
    def pair_spread(pair_id: str) -> Any:
        """Return spread time series + regime overlay for a single pair (spec — Pair Explorer)."""
        import pandas as pd
        with discovery_store.lock:
            result = discovery_store.result
        if result is None:
            return jsonify({"ok": False, "error": "No discovery run yet"}), 404

        pairs_df = result.get("pairs_df")
        price_matrix = result.get("price_matrix")
        regime_series = result.get("regime_series")

        if pairs_df is None or price_matrix is None:
            return jsonify({"ok": False, "error": "Discovery result incomplete"}), 500

        row = pairs_df[pairs_df["pair_id"] == pair_id]
        if row.empty:
            return jsonify({"ok": False, "error": f"Pair {pair_id!r} not found"}), 404

        row = row.iloc[0]
        asset_a = str(row["asset_A"])
        asset_b = str(row["asset_B"])
        hedge_ratio = float(row.get("hedge_ratio", 1.0))

        if asset_a not in price_matrix.columns or asset_b not in price_matrix.columns:
            return jsonify({"ok": False, "error": "Tickers not in price matrix"}), 404

        spread = (price_matrix[asset_a] - hedge_ratio * price_matrix[asset_b]).dropna()
        spread_mean = float(spread.mean())
        spread_std = float(spread.std())
        spread_z = (spread - spread_mean) / spread_std if spread_std > 0 else spread

        spread_data = [
            {"date": str(idx.date()), "spread": round(float(v), 6), "z": round(float(sz), 4)}
            for idx, v, sz in zip(spread.index, spread.values, spread_z.values)
        ]

        regime_data: list = []
        if regime_series is not None:
            aligned = regime_series.reindex(spread.index, method="ffill").dropna()
            regime_data = [
                {"date": str(idx.date()), "regime": int(v)}
                for idx, v in aligned.items()
            ]

        return jsonify({
            "ok": True,
            "pair_id": pair_id,
            "asset_A": asset_a,
            "asset_B": asset_b,
            "hedge_ratio": hedge_ratio,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "spread": spread_data,
            "regime": regime_data,
        })

    @app.get("/api/network")
    def network_graph() -> Any:
        """Return nodes/edges for the pair network graph (spec — Network View).

        Optional query param: ``regime=<int>`` filters edges to a single regime.
        """
        with discovery_store.lock:
            result = discovery_store.result
        if result is None:
            return jsonify({"ok": False, "error": "No discovery run yet"}), 404

        pairs_df = result.get("pairs_df")
        if pairs_df is None or pairs_df.empty:
            return jsonify({"ok": True, "nodes": [], "edges": []})

        regime_param = request.args.get("regime")
        if regime_param is not None:
            try:
                pairs_df = pairs_df[pairs_df["regime"] == int(regime_param)]
            except (ValueError, TypeError):
                pass

        tickers = sorted(set(pairs_df["asset_A"].tolist() + pairs_df["asset_B"].tolist()))
        nodes = [{"id": t, "label": t} for t in tickers]

        agg = (
            pairs_df.groupby("pair_id")
            .agg(asset_A=("asset_A", "first"), asset_B=("asset_B", "first"),
                 best_pvalue=("coint_pvalue", "min"), avg_corr=("corr", "mean"))
            .reset_index()
        )
        edges = [
            {
                "source": r["asset_A"],
                "target": r["asset_B"],
                "pair_id": r["pair_id"],
                "weight": round(float(1 - r["best_pvalue"]), 4),
                "corr": round(float(r["avg_corr"]), 4),
            }
            for _, r in agg.iterrows()
        ]
        return jsonify({"ok": True, "nodes": nodes, "edges": edges})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
