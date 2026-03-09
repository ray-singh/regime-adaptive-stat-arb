from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# Ensure src/ is importable when running from dashboard/backend
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import PlatformConfig, setup_logging
from backtest.run_backtest import run_backtest
from backtest.job_queue import BacktestJobQueue, JobStatus


@dataclass
class BacktestStore:
    """In-memory state for latest dashboard backtest."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    last_error: Optional[str] = None
    last_result: Optional[dict[str, Any]] = None
    cache: dict[str, Any] = field(default_factory=dict)


store = BacktestStore()


def _run_backtest_from_payload(payload: dict) -> dict:
    """Thin adapter so BacktestJobQueue can call run_backtest."""
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
                for _, row in grp.iterrows():
                    qty = float(row.get("quantity") or 0)
                    price = float(row.get("fill_price") or 0)
                    commission = float(row.get("commission") or 0)
                    cash_flow = -qty * price - commission
                    if start_date is None:
                        start_date = row.get("date")
                    running_qty += qty
                    acc_cash += cash_flow
                    legs.append({
                        "date": str(row.get("date")),
                        "ticker": row.get("ticker"),
                        "quantity": qty,
                        "fill_price": price,
                        "commission": commission,
                        "cash_flow": float(cash_flow),
                        "pair_id": pair,
                    })

                    # Record enriched leg (will attach roundtrip id when closed)
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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
