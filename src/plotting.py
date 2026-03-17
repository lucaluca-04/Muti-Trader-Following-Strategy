from __future__ import annotations

"""
Centralised plotting utilities shared across all scripts.

Covers:
- portfolio equity + drawdown  (from backtest_plot.py)
- rolling backtest with rebalance markers  (from profiling_step5_rolling_backtest.py)
- best-trader-per-cluster equity curves  (from profiling_step3_cluster_profiles.py)
- cluster / style-drift summary  (from profiling_visualize_step3_step4.py)
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:
    plt = None  # type: ignore


# ─── Portfolio equity + drawdown ─────────────────────────────────────────────

def compute_equity_and_drawdown(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (equity, drawdown) series from a periodic-return series."""
    equity = (1.0 + returns).cumprod()
    drawdown = (equity - equity.cummax()) / equity.cummax()
    return equity, drawdown


def plot_equity_drawdown(
    returns: pd.Series,
    output_path: str | Path,
    title: str = "Portfolio Equity Curve",
) -> None:
    """Plot equity curve + drawdown and save to output_path."""
    if plt is None:
        return
    equity, drawdown = compute_equity_and_drawdown(returns)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax1.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.5)
    ax1.set_ylabel("Equity (TWR)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Rolling backtest with rebalance markers ─────────────────────────────────

def plot_rolling_backtest(
    equity: pd.Series,
    full_ret: pd.Series,
    weight_rows: list[dict],
    output_path: str | Path,
) -> None:
    """Plot equity curve, drawdown, and rebalance date markers."""
    if plt is None:
        return
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, height_ratios=[2, 1]
    )

    ax1.plot(equity.index, equity.values, color="steelblue", linewidth=1.5,
             label="Portfolio equity (TWR)")
    ax1.set_ylabel("Equity (TWR)")
    ax1.set_title("Rolling backtest: core-satellite (no look-ahead)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    reb_dates = sorted(set(r["rebalance_date"] for r in weight_rows))
    for i, d in enumerate(reb_dates):
        ax1.axvline(
            x=pd.Timestamp(d), color="gray", linestyle="--", alpha=0.7,
            label="Rebalance" if i == 0 else None,
        )

    _, drawdown = compute_equity_and_drawdown(full_ret)
    ax2.fill_between(equity.index, drawdown.reindex(equity.index).values, 0,
                     color="coral", alpha=0.5, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── Best trader per cluster equity curves ───────────────────────────────────

def plot_best_trader_curves(
    df: pd.DataFrame,
    curves_dir: Path,
    output_path: str | Path,
    output_dir: Path | None = None,
) -> None:
    """
    Plot equity curve of best trader per cluster (all on one figure).
    Optionally save one PNG per cluster into output_dir.
    """
    if plt is None:
        return
    from src.dataio import load_equity_curve
    from src.profiling import best_trader_per_cluster

    clusters_sorted = sorted([c for c in df["cluster"].unique() if c != -1])
    if not clusters_sorted:
        return

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for cl in clusters_sorted:
            trader = best_trader_per_cluster(df, cl)
            if trader is None:
                continue
            curve = load_equity_curve(curves_dir, trader)
            if curve is None or curve.empty:
                continue
            row = df[df["trader"] == trader].iloc[0]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(curve["date"], curve["equity_twr"], color="steelblue", linewidth=1.5)
            ax.set_title(f"Cluster {cl} – best trader (by ex-ante Sharpe)\n{trader}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity (TWR)")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(output_dir / f"cluster_{cl}.png", dpi=120)
            plt.close()

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters_sorted), 1)))
    for i, cl in enumerate(clusters_sorted):
        trader = best_trader_per_cluster(df, cl)
        if trader is None:
            continue
        curve = load_equity_curve(curves_dir, trader)
        if curve is None or curve.empty:
            continue
        row = df[df["trader"] == trader].iloc[0]
        mean_r = float(row.get("mean_return", np.nan))
        std_r = float(row.get("std_return", np.nan))
        max_dd = float(row.get("max_drawdown_synth", np.nan))
        sharpe = mean_r / std_r if std_r > 0 else np.nan
        total_ret = float(curve["equity_twr"].iloc[-1] - 1.0)
        label = (
            f"Cluster {cl}: mean_r={mean_r:.4f}, std={std_r:.4f}, "
            f"max_dd={max_dd:.4f}, Sharpe={sharpe:.3f}, total_ret={total_ret:.2%}"
        )
        ax.plot(curve["date"], curve["equity_twr"],
                color=colors[i % len(colors)], label=label, alpha=0.9)

    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (TWR)")
    ax.set_title("Best trader per cluster (by ex-ante Sharpe) – equity curves")
    ax.legend(loc="upper left", fontsize=6, ncol=1,
              bbox_to_anchor=(1.02, 1), frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── Cluster / style-drift summary ───────────────────────────────────────────

def plot_cluster_drift_summary(
    clusters_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """2×2 summary figure: cluster distribution, drift score histogram, drift by type, windows."""
    if plt is None:
        return
    merged = drift_df.merge(clusters_df, on="trader", how="inner")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    counts = clusters_df["cluster"].value_counts().sort_index()
    colors = ["#888888" if c == -1 else "steelblue" for c in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Step3: Cluster assignment (HDBSCAN)")
    ax.set_xlabel("Cluster (-1 = noise)")
    ax.set_ylabel("Number of traders")
    ax.tick_params(axis="x", rotation=0)

    ax = axes[0, 1]
    drift_df["style_drift_score"].hist(bins=20, ax=ax, edgecolor="black", alpha=0.7)
    ax.set_title("Step4: Style drift score")
    ax.set_xlabel("style_drift_score (0=stable, 1=unstable/noise)")
    ax.set_ylabel("Number of traders")

    ax = axes[1, 0]
    merged["is_noise"] = merged["cluster"] == -1
    merged.boxplot(column="style_drift_score", by="is_noise", ax=ax)
    ax.set_title("Style drift by cluster type")
    ax.set_xlabel("In cluster (False = noise)")
    ax.set_ylabel("style_drift_score")
    plt.suptitle("")

    ax = axes[1, 1]
    vc = drift_df["num_valid_windows"].value_counts().sort_index()
    vc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Count of traders by num_valid_windows")
    ax.set_xlabel("num_valid_windows")
    ax.set_ylabel("Number of traders")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
