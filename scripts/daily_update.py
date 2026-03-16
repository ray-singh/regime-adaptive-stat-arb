"""Daily end-of-day pipeline update.
Run via cron at 7pm ET Mon-Fri after market close.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure src/ is on the path when run as a script
ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import PlatformConfig, setup_logging
from data.yfinance_client import YFinanceClient
from features.featurize import compute_market_features
from regime.hmm_detector import HMMRegimeDetector
from pair_discovery import PairDiscoveryEngine
from relationship_analysis import RelationshipAnalyzer
from pair_ranking import PairRankingEngine

setup_logging(level="INFO", log_file=str(ROOT / "logs" / "daily_update.log"))
logger = logging.getLogger("daily_update")


def run():
    start = datetime.utcnow()
    logger.info("=== Daily update started at %s UTC ===", start.isoformat())

    cfg = PlatformConfig()
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch fresh prices ────────────────────────────────────
    logger.info("Step 1/5: Fetching price data for %d tickers", len(cfg.data.tickers))
    client = YFinanceClient(cache_dir=str(ROOT / "data" / "cache"))
    price_matrix = client.get_price_matrix(
        cfg.data.tickers,
        start_date="2015-01-01",   # full history; yfinance returns cached+new
        end_date=None,
    )
    if price_matrix.empty:
        logger.error("Price matrix is empty — aborting update.")
        sys.exit(1)
    logger.info("Price matrix: %d rows x %d cols, last date: %s",
                len(price_matrix), len(price_matrix.columns),
                price_matrix.index[-1].date())

    # ── Step 2: Recompute market features ────────────────────────────
    logger.info("Step 2/5: Computing market features")
    features = compute_market_features(price_matrix)
    logger.info("Features computed: %d rows", len(features))

    # ── Step 3: Re-run HMM regime detection ──────────────────────────
    logger.info("Step 3/5: Running walk-forward HMM regime detection")
    hmm_cols = ["avg_rv", "ret_dispersion", "index_momentum"]
    detector = HMMRegimeDetector(
        n_states=cfg.regime.n_states,
        feature_cols=hmm_cols,
    )
    feature_df = features[hmm_cols].dropna()
    regime_series = detector.fit_predict_walkforward(feature_df, min_train_years=2)
    regime_series = regime_series.dropna().astype(int)
    logger.info("Regime series: %d labels, distribution: %s",
                len(regime_series),
                regime_series.value_counts().sort_index().to_dict())

    # ── Step 4: Re-run pair discovery ────────────────────────────────
    logger.info("Step 4/5: Running pair discovery")
    engine = PairDiscoveryEngine(min_regime_bars=126)
    pairs_df = engine.discover(price_matrix, regime_series, tickers=cfg.data.tickers)

    if pairs_df.empty:
        logger.warning("No pairs discovered — cache will not be updated.")
        sys.exit(1)

    analyzer = RelationshipAnalyzer(price_matrix=price_matrix)
    pair_regime_stats, pair_summary = analyzer.analyse(pairs_df, regime_series)

    ranker = PairRankingEngine()
    ranked = ranker.rank(pair_summary)
    logger.info("Discovery complete: %d pairs, %d unique",
                len(pairs_df), pairs_df["pair_id"].nunique())

    # ── Step 5: Write results and invalidate cache ────────────────────
    logger.info("Step 5/5: Persisting results and clearing stale cache")
    import pickle, time

    results = {
        "pairs_df":         pairs_df,
        "pair_regime_stats": pair_regime_stats,
        "pair_summary":     pair_summary,
        "ranked":           ranked,
        "regime_series":    regime_series,
        "price_matrix":     price_matrix,
        "market_features":  features,
        "updated_at":       datetime.utcnow().isoformat(),
    }

    # Write the canonical result file the Flask app reads on startup
    result_path = cache_dir / "latest_discovery.pkl"
    with open(result_path, "wb") as f:
        pickle.dump({"timestamp": time.time(), "data": results}, f)
    logger.info("Wrote latest_discovery.pkl")

    # Delete all stale hmm_*, pairs_*, ranking_* cache files so the
    # Flask app recomputes fresh on next /api/discover request
    deleted = 0
    for prefix in ("hmm_", "pairs_", "ranking_"):
        for stale in cache_dir.glob(f"{prefix}*.pkl"):
            stale.unlink()
            deleted += 1
    logger.info("Cleared %d stale cache files", deleted)

    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info("=== Update complete in %.1fs ===", elapsed)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        logging.getLogger("daily_update").exception(
            "Daily update FAILED: %s", exc
        )
        sys.exit(1)