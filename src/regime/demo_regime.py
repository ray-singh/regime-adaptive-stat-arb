"""Demo: Regime Detection + Pairs Trading Evaluation.

Usage (from /src):
    python -m regime.demo_regime

Uses cached OHLCV data; downloads if missing.
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns

# Ensure imports resolve from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yfinance_client import YFinanceClient
from features.featurize import compute_standard_features
from regime.hmm_detector import HMMRegimeDetector
from regime.volatility_detector import VolatilityRegimeDetector
from regime.clustering_detector import ClusteringRegimeDetector
from strategy.pairs_trading import PairsSelector, PairsTradingStrategy, build_price_matrix

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                "TSLA", "JPM", "V", "UNH", "ADBE", "AMD", "AVGO", "CRM", "ORCL"]
REGIME_TICKER = "AAPL"     # use AAPL to showcase regime detection
PERIOD        = "10y"
INTERVAL      = "1d"
PLOTS_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "..", "data", "plots")

REGIME_COLOR  = {0: "#2ca02c", 1: "#ff7f0e", 2: "#d62728", 3: "#9467bd"}
REGIME_NAME   = {0: "Bull/Low-Vol", 1: "Neutral", 2: "Bear/High-Vol", 3: "Crisis"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def fetch_features(tickers, period, interval):
    client = YFinanceClient()
    features = {}
    for t in tickers:
        try:
            raw = client.fetch_ticker(t, period=period, interval=interval, use_cache=True)
            feat = compute_standard_features(raw)
            feat.index = pd.to_datetime(feat["Date"]).dt.tz_localize(None)
            features[t] = feat
        except Exception as e:
            logger.warning(f"Could not fetch {t}: {e}")
    return features


def background_regimes(ax, labels: pd.Series, alpha=0.12):
    """Shade background of ax by regime colour."""
    prev_r, prev_d = None, None
    for d, r in labels.items():
        if r != prev_r:
            if prev_r is not None:
                ax.axvspan(prev_d, d, alpha=alpha, color=REGIME_COLOR.get(prev_r, "grey"),
                           linewidth=0)
            prev_r, prev_d = r, d
    if prev_r is not None:
        ax.axvspan(prev_d, labels.index[-1], alpha=alpha,
                   color=REGIME_COLOR.get(prev_r, "grey"), linewidth=0)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Regime Detection
# ──────────────────────────────────────────────────────────────────────────────

def run_regime_detection(features: dict) -> dict:
    feat_df = features[REGIME_TICKER]

    detectors = {
        "HMM (3-state)":         HMMRegimeDetector(n_states=3),
        "Volatility (3-state)":  VolatilityRegimeDetector(n_states=3),
        "K-Means (3-state)":     ClusteringRegimeDetector(n_states=3),
    }

    results = {}
    for name, det in detectors.items():
        labels = det.fit_predict(feat_df)
        summary = det.regime_summary(labels)
        stats   = det.state_stats(feat_df, labels)
        results[name] = {"labels": labels, "summary": summary, "stats": stats, "detector": det}

        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")
        print("Regime distribution:\n", summary.to_string())
        print("\nState stats:\n", stats.to_string())

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Regime Agreement Matrix
# ──────────────────────────────────────────────────────────────────────────────

def regime_agreement(results: dict) -> pd.DataFrame:
    hmm_l  = results["HMM (3-state)"]["labels"]
    vol_l  = results["Volatility (3-state)"]["labels"]
    km_l   = results["K-Means (3-state)"]["labels"]

    common = hmm_l.index.intersection(vol_l.index).intersection(km_l.index)
    df = pd.DataFrame({"HMM": hmm_l[common], "Vol": vol_l[common], "KMeans": km_l[common]})

    # pairwise % agreement
    pairs = [("HMM", "Vol"), ("HMM", "KMeans"), ("Vol", "KMeans")]
    print("\n\nRegime Agreement (% same state):")
    for a, b in pairs:
        pct = (df[a] == df[b]).mean() * 100
        print(f"  {a} vs {b}: {pct:.1f}%")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Pairs Trading
# ──────────────────────────────────────────────────────────────────────────────

def run_pairs_trading(features: dict) -> dict:
    # Build price matrix
    frames = []
    for t, fdf in features.items():
        price_col = "adj_close" if "adj_close" in fdf.columns else "close"
        s = fdf[price_col].rename(t)
        s.index = pd.to_datetime(fdf["Date"]).dt.tz_localize(None) if "Date" in fdf.columns else fdf.index
        frames.append(s)

    price_matrix = pd.concat(frames, axis=1).sort_index()

    # Cointegration scan
    selector = PairsSelector(pvalue_threshold=0.05, min_half_life=5, max_half_life=126)
    cointegrated = selector.find_pairs(price_matrix, verbose=False)

    print(f"\n{'='*60}")
    print(f" Cointegration Results — {len(cointegrated)} pairs found")
    print(f"{'='*60}")
    if not cointegrated.empty:
        print(cointegrated.to_string(index=False))

    # Trade top pairs
    strategy = PairsTradingStrategy(zscore_window=60, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    traded = {}

    for _, row in cointegrated.head(5).iterrows():
        t1, t2, hr = row["ticker1"], row["ticker2"], row["hedge_ratio"]
        sig = strategy.generate_signals(price_matrix, t1, t2, hr)
        ev  = strategy.evaluate(sig)
        traded[f"{t1}/{t2}"] = {"signals": sig, "eval": ev}

        print(f"\n  Pair: {t1}/{t2}  (hedge_ratio={hr:.3f}, HL={row['half_life_days']:.0f}d)")
        for k, v in ev.items():
            print(f"    {k}: {v}")

    return {"cointegrated": cointegrated, "traded": traded, "price_matrix": price_matrix}


# ──────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_regimes(features, results):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    feat_df = features[REGIME_TICKER]
    price_col = "adj_close" if "adj_close" in feat_df.columns else "close"

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f"Regime Detection — {REGIME_TICKER}", fontsize=15, fontweight="bold")

    # Price
    ax_price = axes[0]
    ax_price.plot(feat_df.index, feat_df[price_col], lw=0.8, color="steelblue")
    ax_price.set_ylabel("Price ($)")
    ax_price.set_title("Price with HMM Regimes")
    background_regimes(ax_price, results["HMM (3-state)"]["labels"])
    patches = [mpatches.Patch(color=REGIME_COLOR[r], label=REGIME_NAME[r], alpha=0.6)
               for r in sorted(REGIME_COLOR)]
    ax_price.legend(handles=patches, loc="upper left", fontsize=8)

    # Realized vol comparison
    ax_rv = axes[1]
    ax_rv.plot(feat_df.index, feat_df["rv_20"], lw=0.8, label="rv_20")
    ax_rv.set_ylabel("Realized Vol")
    ax_rv.set_title("Realized Vol (20d) with Vol-Threshold Regimes")
    background_regimes(ax_rv, results["Volatility (3-state)"]["labels"])

    # Regime colours (all 3 detectors as heatmap rows)
    ax_cmp = axes[2]
    names = list(results.keys())
    data = np.vstack([results[n]["labels"].reindex(feat_df.index).ffill().bfill().values
                      for n in names])
    im = ax_cmp.imshow(data, aspect="auto", cmap="RdYlGn_r",
                       extent=[0, data.shape[1], 0, len(names)],
                       vmin=0, vmax=3, interpolation="nearest")
    ax_cmp.set_yticks([0.5, 1.5, 2.5])
    ax_cmp.set_yticklabels(["K-Means", "Vol", "HMM"])
    ax_cmp.set_title("Regime Labels Comparison")
    fig.colorbar(im, ax=ax_cmp, orientation="vertical", pad=0.01, fraction=0.02,
                 ticks=[0, 1, 2, 3])

    # Z-score plot for top pair (if any)
    ax_z = axes[3]
    ax_z.set_title("(No cointegrated pairs found)" )
    ax_z.axis("off")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "regime_detection.png")
    plt.savefig(out, dpi=150)
    print(f"\nRegime plot saved → {out}")
    plt.close(fig)


def plot_pairs(pairs_result):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    traded = pairs_result["traded"]
    if not traded:
        print("No pairs to plot.")
        return

    n = min(3, len(traded))
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    for idx, (pair_name, p) in enumerate(list(traded.items())[:n]):
        sig = p["signals"]
        ax_z, ax_pnl = axes[idx]

        # Z-score + signals
        ax_z.plot(sig.index, sig["zscore"], lw=0.7, color="steelblue")
        ax_z.axhline(2.0, ls="--", color="red", lw=0.8)
        ax_z.axhline(-2.0, ls="--", color="green", lw=0.8)
        ax_z.axhline(0.5, ls=":", color="grey", lw=0.6)
        ax_z.axhline(-0.5, ls=":", color="grey", lw=0.6)
        long_entries  = sig[(sig["position"] == 1) & (sig["position"].shift(1) == 0)]
        short_entries = sig[(sig["position"] == -1) & (sig["position"].shift(1) == 0)]
        ax_z.scatter(long_entries.index,  long_entries["zscore"],  marker="^", color="green", s=20, zorder=3)
        ax_z.scatter(short_entries.index, short_entries["zscore"], marker="v", color="red",   s=20, zorder=3)
        ax_z.set_title(f"{pair_name} — Z-Score")
        ax_z.set_ylabel("Z-score")

        # Cumulative P&L
        ax_pnl.plot(sig.index, sig["pnl_cumulative"] * 100, lw=0.8, color="darkgreen")
        ax_pnl.axhline(0, ls="--", color="grey", lw=0.6)
        ev = p["eval"]
        ax_pnl.set_title(
            f"{pair_name} — Cum P&L | Sharpe={ev['sharpe_ratio']:.2f} | MaxDD={ev['max_drawdown_pct']:.1f}%"
        )
        ax_pnl.set_ylabel("Return (%)")

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "pairs_trading.png")
    plt.savefig(out, dpi=150)
    print(f"Pairs plot saved → {out}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" Regime-Adaptive Stat-Arb Demo")
    print("=" * 60)

    print(f"\n[1/4] Fetching features for {len(DEMO_TICKERS)} tickers…")
    features = fetch_features(DEMO_TICKERS, PERIOD, INTERVAL)
    print(f"  → Loaded: {list(features.keys())}")

    print(f"\n[2/4] Running regime detection on {REGIME_TICKER}…")
    regime_results = run_regime_detection(features)

    print("\n[3/4] Agreement analysis…")
    regime_agreement(regime_results)

    print("\n[4/4] Pairs trading…")
    pairs_result = run_pairs_trading(features)

    # Plots
    print("\n[Plots] Generating…")
    plot_regimes(features, regime_results)
    plot_pairs(pairs_result)

    print("\nDone!")


if __name__ == "__main__":
    main()
