"""Pairs Trading Strategy.

Workflow:
    1. PairsSelector.find_pairs() — Engle-Granger cointegration test on all pairs
    2. PairsTradingStrategy.generate_signals() — rolling z-score of spread
    3. Evaluate() — basic P&L, Sharpe, hit-rate per regime

Signal rules:
    Entry long spread  (buy leg1, sell leg2): z < -entry_z
    Entry short spread (sell leg1, buy leg2): z >  entry_z
    Exit: |z| <  exit_z
    Stop: |z| >  stop_z
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from typing import List, Optional, Tuple, Dict
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import atexit

from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker for parallel cointegration testing (must be picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _test_pair_worker(args: tuple):
    """Run Engle-Granger test + hedge-ratio + half-life for one pair.
    Returns a result dict on success, or None if the pair is rejected."""
    t1, t2, s1_vals, s2_vals, pvalue_threshold, min_hl, max_hl = args
    import warnings
    import numpy as np
    from statsmodels.tsa.stattools import coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            _, pvalue, _ = coint(s1_vals, s2_vals)
        except Exception:
            return None

    if pvalue > pvalue_threshold:
        return None

    # OLS hedge ratio: s1 = β·s2 + α
    X = add_constant(s2_vals)
    res = OLS(s1_vals, X).fit()
    hedge_ratio = float(res.params[1])

    spread = s1_vals - hedge_ratio * s2_vals

    # Ornstein-Uhlenbeck half-life via AR(1)
    lagged = spread[:-1]
    delta = np.diff(spread)
    X2 = add_constant(lagged)
    res2 = OLS(delta, X2).fit()
    theta = -res2.params[1]
    if theta <= 0:
        return None
    half_life = float(np.log(2) / theta)

    if not (min_hl <= half_life <= max_hl):
        return None

    return {
        "ticker1": t1,
        "ticker2": t2,
        "pvalue": round(float(pvalue), 5),
        "hedge_ratio": round(hedge_ratio, 4),
        "half_life_days": round(half_life, 1),
        "spread_mean": round(float(spread.mean()), 4),
        "spread_std": round(float(spread.std()), 4),
    }


def _test_pairs_batch_worker(batch: list):
    """Worker that receives a list of args tuples and returns list of results.
    This reduces IPC overhead by batching several pair tests per process.
    """
    results = []
    for args in batch:
        try:
            res = _test_pair_worker(args)
            if res is not None:
                results.append(res)
        except Exception:
            # don't let one bad pair kill the whole batch
            continue
    return results


# Module-level executor reused across calls to avoid repeated process spawn
_GLOBAL_EXECUTOR: ProcessPoolExecutor | None = None
_GLOBAL_EXECUTOR_WORKERS: int | None = None


def _get_executor(max_workers: int) -> ProcessPoolExecutor:
    global _GLOBAL_EXECUTOR, _GLOBAL_EXECUTOR_WORKERS
    # Recreate pool if worker count changed to avoid stale sizing.
    if _GLOBAL_EXECUTOR is not None and _GLOBAL_EXECUTOR_WORKERS != max_workers:
        _GLOBAL_EXECUTOR.shutdown(wait=False)
        _GLOBAL_EXECUTOR = None
        _GLOBAL_EXECUTOR_WORKERS = None

    if _GLOBAL_EXECUTOR is None:
        # Try to prefer 'fork' start method on platforms that support it to
        # reduce pickling overhead; fall back to default when unavailable.
        try:
            ctx = multiprocessing.get_context("fork")
        except Exception:
            ctx = None

        if ctx is not None:
            _GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)  # type: ignore[arg-type]
        else:
            _GLOBAL_EXECUTOR = ProcessPoolExecutor(max_workers=max_workers)
        _GLOBAL_EXECUTOR_WORKERS = max_workers

        # Ensure executor is cleaned up on interpreter exit
        atexit.register(lambda: _GLOBAL_EXECUTOR.shutdown(wait=False) if _GLOBAL_EXECUTOR is not None else None)
    return _GLOBAL_EXECUTOR


# ─────────────────────────────────────────────────────────────────────────────
# Pair Selection
# ─────────────────────────────────────────────────────────────────────────────

class PairsSelector:
    """Find cointegrated pairs using Engle-Granger test.

    Parameters
    ----------
    pvalue_threshold : float
        Maximum p-value for cointegration (default 0.05).
    min_half_life : int
        Minimum mean-reversion half-life in days.
    max_half_life : int
        Maximum mean-reversion half-life in days.
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        min_half_life: int = 5,
        max_half_life: int = 126,
    ):
        self.pvalue_threshold = pvalue_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def find_pairs(
        self,
        price_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        max_pairs: int = 50,
        verbose: bool = True,
        sequential_threshold: int = 64,
    ) -> pd.DataFrame:
        """Find cointegrated pairs in a wide price DataFrame.

        Parameters
        ----------
        price_df : DataFrame
            Wide price matrix (index=Date, columns=tickers, values=adj_close/close).
        tickers : list[str], optional
            Subset of tickers to test. Default = all columns.
        max_pairs : int
            Maximum pairs to return (ranked by p-value).

        Returns
        -------
        DataFrame with columns: ticker1, ticker2, pvalue, hedge_ratio, half_life_days
        """
        tickers = tickers or list(price_df.columns)

        combos = list(combinations(tickers, 2))
        logger.info("Testing %d raw pairs for cointegration...", len(combos))

        # ── Fast pre-filter: correlation ≥ 0.7 required (much cheaper than coint)
        corr = price_df[tickers].corr()
        combos = [
            (t1, t2) for (t1, t2) in combos
            if abs(corr.at[t1, t2]) >= 0.70
        ]
        logger.info("  After |r|≥0.70 pre-filter: %d pairs remain", len(combos))

        # ── Build argument list (convert to numpy arrays once, outside workers)
        # Cache cleaned per-ticker series to avoid repeated dropna/index ops.
        cleaned = {t: price_df[t].dropna() for t in tickers}
        args_list: list = []
        for t1, t2 in combos:
            s1 = cleaned[t1]
            s2 = cleaned[t2]
            common = s1.index.intersection(s2.index)
            # Require a reasonable amount of overlap between legs before testing.
            # For global discovery 252 trading days (~1 year) is a strict default,
            # but for per-regime discovery regimes may be shorter. Use a dynamic
            # threshold: at least 60 days OR 60% of the regime slice, capped at 252.
            min_common = max(60, int(min(252, len(price_df) * 0.6)))
            if len(common) < min_common:
                continue
            args_list.append((
                t1, t2,
                s1[common].to_numpy(dtype=float),
                s2[common].to_numpy(dtype=float),
                self.pvalue_threshold,
                self.min_half_life,
                self.max_half_life,
            ))

        # ── Parallel execution via reusable ProcessPoolExecutor with batching
        # Use at most 4 workers (avoid over-subscribing on small universes)
        n_workers = max(1, min(os.cpu_count() or 4, 4, (len(args_list) + 3) // 4))
        pairs_result: list = []
        if not args_list:
            return pd.DataFrame(pairs_result)

        # For smaller candidate sets, process start + IPC overhead dominates.
        if len(args_list) < sequential_threshold:
            for a in args_list:
                r = _test_pair_worker(a)
                if r is not None:
                    pairs_result.append(r)
            result = pd.DataFrame(pairs_result)
            if not result.empty:
                result = result.sort_values("pvalue").head(max_pairs).reset_index(drop=True)
            logger.info("Found %d cointegrated pairs", len(result))
            return result

        # Batch size: aim for ~8 batches per worker to amortize scheduling.
        # Cap at 128 to avoid very large task payloads.
        batch_size = min(128, max(1, len(args_list) // (n_workers * 8)))
        batches = [args_list[i:i + batch_size] for i in range(0, len(args_list), batch_size)]

        try:
            exe = _get_executor(n_workers)
            # Use map to submit batches; each worker runs multiple tests inside
            for batch_res in exe.map(_test_pairs_batch_worker, batches):
                if not batch_res:
                    continue
                pairs_result.extend(batch_res)
        except Exception as exc:
            # Fallback to sequential execution if multiprocessing fails
            logger.warning("Batched parallel pair testing failed (%s); falling back to sequential.", exc)
            for a in args_list:
                r = _test_pair_worker(a)
                if r is not None:
                    pairs_result.append(r)

        if verbose:
            logger.info("  %d cointegrated pairs found", len(pairs_result))

        result = pd.DataFrame(pairs_result)
        if not result.empty:
            result = (
                result.sort_values("pvalue")
                .head(max_pairs)
                .reset_index(drop=True)
            )
        logger.info("Found %d cointegrated pairs", len(result))
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_hedge_ratio(s1: pd.Series, s2: pd.Series) -> float:
        """OLS hedge ratio: s1 = β * s2 + α."""
        X = add_constant(s2.values)
        res = OLS(s1.values, X).fit()
        return float(res.params[1])

    @staticmethod
    def _half_life(spread: pd.Series) -> float:
        """Ornstein-Uhlenbeck half-life via AR(1) regression."""
        lagged = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        common = lagged.index.intersection(delta.index)
        lagged, delta = lagged[common], delta[common]
        X = add_constant(lagged.values)
        res = OLS(delta.values, X).fit()
        theta = -res.params[1]
        if theta <= 0:
            return np.inf
        return np.log(2) / theta


# ─────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ─────────────────────────────────────────────────────────────────────────────

class PairsTradingStrategy:
    """Generate long/short spread signals via rolling z-score.

    Parameters
    ----------
    zscore_window : int
        Rolling window for z-score computation.
    entry_z : float
        |z| threshold to open a position.
    exit_z : float
        |z| below which to close.
    stop_z : float
        |z| above which to stop-loss.
    """

    def __init__(
        self,
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ):
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def compute_spread(
        self,
        price_df: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        hedge_ratio: float,
    ) -> pd.Series:
        """Spread = price1 - hedge_ratio * price2."""
        s1 = price_df[ticker1]
        s2 = price_df[ticker2]
        return (s1 - hedge_ratio * s2).rename(f"{ticker1}/{ticker2}")

    def compute_zscore(self, spread: pd.Series) -> pd.Series:
        mu = spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2).mean()
        sigma = spread.rolling(self.zscore_window, min_periods=self.zscore_window // 2).std()
        return ((spread - mu) / sigma.replace(0, np.nan)).rename("zscore")

    def generate_signals(
        self,
        price_df: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        hedge_ratio: float,
        regime_labels: Optional[pd.Series] = None,
        active_regimes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Generate daily signals for a single pair.

        Returns a DataFrame with columns:
            spread, zscore, position, pnl_daily, pnl_cumulative
        where position is:
            +1 = long spread (buy t1, sell t2)
            -1 = short spread (sell t1, buy t2)
             0 = flat
        """
        spread = self.compute_spread(price_df, ticker1, ticker2, hedge_ratio)
        zscore = self.compute_zscore(spread)

        # Raw returns of each leg
        ret1 = price_df[ticker1].pct_change()
        ret2 = price_df[ticker2].pct_change()

        # ── Vectorized state machine ─────────────────────────────────────────
        # Convert to numpy *before* the loop so each bar uses O(1) array access
        # instead of pandas .iloc[], giving a 10-30x speedup on long histories.
        z_arr = zscore.to_numpy(dtype=float)
        n = len(z_arr)
        pos_arr = np.zeros(n, dtype=np.int8)

        # Pre-compute regime membership mask outside the loop (vectorised reindex)
        if regime_labels is not None and active_regimes is not None:
            active_set = frozenset(active_regimes)
            rl_vals = regime_labels.reindex(zscore.index).to_numpy()
            # True = entry allowed (unknown/NaN regime also allows entry)
            regime_ok = np.array(
                [True if (v != v or v in active_set) else False for v in rl_vals],
                dtype=bool,
            )
        else:
            regime_ok = np.ones(n, dtype=bool)

        # Scalar thresholds (local vars avoid repeated attribute lookup in loop)
        entry_z = self.entry_z
        exit_z = self.exit_z
        stop_z = self.stop_z

        pos = 0
        for i in range(1, n):
            z = z_arr[i]
            if z != z:  # NaN check — faster than np.isnan / pd.isna in tight loops
                pos = 0
                continue

            # Exit / stop (evaluated regardless of regime — never trap a position)
            if pos != 0:
                az = z if z >= 0.0 else -z  # inline abs to avoid Python overhead
                if az < exit_z or az > stop_z:
                    pos = 0

            # Entry (gated by regime)
            if pos == 0 and regime_ok[i]:
                if z < -entry_z:
                    pos = 1
                elif z > entry_z:
                    pos = -1

            pos_arr[i] = pos

        position = pd.Series(pos_arr.astype(int), index=zscore.index)

        # P&L: long spread = long t1 + short t2, scaled by hedge ratio
        pnl = position.shift(1) * (ret1 - hedge_ratio * ret2)

        result = pd.DataFrame({
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "pnl_daily": pnl,
            "pnl_cumulative": pnl.cumsum(),
        })
        return result

    def evaluate(self, signals_df: pd.DataFrame, risk_free_rate: float = 0.05) -> Dict:
        """Compute summary statistics for a signals DataFrame."""
        pnl = signals_df["pnl_daily"].dropna()
        cum = signals_df["pnl_cumulative"].dropna()

        n_trades = (signals_df["position"].diff().abs() > 0).sum()
        active_days = (signals_df["position"] != 0).sum()

        daily_rf = risk_free_rate / 252
        excess = pnl - daily_rf
        sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else np.nan

        # Max drawdown
        roll_max = cum.cummax()
        dd = cum - roll_max
        max_dd = dd.min()

        # Win rate on active days
        wins = (pnl[signals_df["position"].shift(1) != 0] > 0).mean()

        return {
            "total_return_pct": round(cum.iloc[-1] * 100, 2) if len(cum) else np.nan,
            "ann_return_pct": round(pnl.mean() * 252 * 100, 2),
            "ann_vol_pct": round(pnl.std() * np.sqrt(252) * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "n_trades": int(n_trades),
            "active_days": int(active_days),
            "win_rate": round(wins, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a wide price matrix from long-format OHLCV DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def build_price_matrix(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Pivot long-format OHLCV df → wide (Date × ticker) price matrix."""
    if price_col not in df.columns:
        price_col = "close"
    # Ensure Date is tz-naive
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    wide = df.pivot(index="Date", columns="ticker", values=price_col)
    return wide.sort_index()


def profile_find_pairs(
    selector: PairsSelector,
    price_df: pd.DataFrame,
    profiler: str = "cprofile",
    **kwargs,
) -> None:
    """Run `selector.find_pairs(price_df, **kwargs)` under cProfile and
    write stats to `profiles/pairs_find.prof` for offline inspection.
    """
    try:
        if profiler == "pyinstrument":
            try:
                from pyinstrument import Profiler
            except Exception as exc:
                logger.warning("pyinstrument not available: %s", exc)
                return

            import os as _os
            out_dir = _os.path.join(_os.path.dirname(__file__), "..", "data", "cache")
            os.makedirs(out_dir, exist_ok=True)
            out_path = _os.path.join(out_dir, "pairs_find_pyinstrument.html")

            profiler_obj = Profiler()
            profiler_obj.start()
            selector.find_pairs(price_df, **kwargs)
            profiler_obj.stop()
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(profiler_obj.output_html())
            logger.info("Wrote pyinstrument profile to %s", out_path)
            return

        import cProfile
        import pstats
        import os as _os

        out_dir = _os.path.join(_os.path.dirname(__file__), "..", "data", "cache")
        os.makedirs(out_dir, exist_ok=True)
        out_path = _os.path.join(out_dir, "pairs_find.prof")

        pr = cProfile.Profile()
        pr.enable()
        selector.find_pairs(price_df, **kwargs)
        pr.disable()
        ps = pstats.Stats(pr).sort_stats("cumtime")
        ps.dump_stats(out_path)
        logger.info("Wrote profile to %s", out_path)
    except Exception as exc:
        logger.warning("Profiling failed: %s", exc)
