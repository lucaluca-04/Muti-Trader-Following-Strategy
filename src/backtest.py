from __future__ import annotations

"""
Rolling core-satellite backtest (no look-ahead).

This module exposes the full backtest as a callable function
rolling_core_satellite_backtest() so that scripts and notebooks can
invoke it without duplicating logic.

All per-window feature computation delegates to src.features,
and clustering/risk-parity helpers delegate to src.profiling.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.dataio import load_all_curves
from src.features import compute_features_from_equity_slice
from src.profiling import (
    FEATURE_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    equal_risk_contribution_weights,
    capacity_cap_from_corr,
    run_hdbscan,
)

# ─── Thresholds (override via rolling_core_satellite_backtest kwargs) ─────────

CORE_STD_THRESHOLD = 0.055   # clusters with median std_return <= this are "Core"
MAX_DD_EXCLUDE = 0.95        # exclude clusters with median max_drawdown_synth >= this
DEFAULT_CORE_RATIO = 0.80


# ─── Rolling feature computation ─────────────────────────────────────────────

def rolling_features_for_cutoff(
    curves: dict[str, pd.Series],
    cutoff: pd.Timestamp,
    lookback_days: int,
    min_days: int,
    window_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Slice each trader's equity to (start, cutoff] and compute features.
    - If window_start is set, use (window_start, cutoff] (expanding window).
    - Otherwise use (cutoff - lookback_days, cutoff].
    Drops traders with fewer than min_days observations.
    """
    start = (
        window_start
        if window_start is not None
        else (cutoff - pd.Timedelta(days=lookback_days))
    )
    rows = []
    for trader, eq in curves.items():
        window = eq[(eq.index > start) & (eq.index <= cutoff)]
        if len(window) < min_days:
            continue
        feats = compute_features_from_equity_slice(window)
        feats["trader"] = trader
        rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=["trader"] + FEATURE_COLUMNS)

    df = pd.DataFrame(rows).set_index("trader")
    df = df[df["std_return"].notna() & (df["std_return"] > 1e-8)]
    return df


# ─── Cluster helpers ──────────────────────────────────────────────────────────

def best_trader_in_cluster(
    feats: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    min_active_days: int = 0,
) -> str | None:
    """Highest ex-ante Sharpe trader in cluster_id, with optional active-days filter."""
    mask = labels == cluster_id
    if not mask.any():
        return None
    g = feats.iloc[mask].copy()
    if min_active_days > 0 and "num_active_days" in g.columns:
        g_active = g[g["num_active_days"] >= min_active_days]
        if not g_active.empty:
            g = g_active
    g["sharpe"] = g["mean_return"] / g["std_return"].replace(0, np.nan)
    valid = g.dropna(subset=["sharpe"])
    if not valid.empty:
        return valid["sharpe"].idxmax()
    return g["mean_return"].idxmax()


def top_traders_in_cluster(
    feats: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    top_k: int,
    min_active_days: int = 0,
) -> list[str]:
    """
    Return up to top_k traders in the cluster ranked by Sharpe, with an optional
    active-days filter. Falls back to mean_return if Sharpe is unavailable.
    """
    if top_k <= 0:
        return []
    mask = labels == cluster_id
    if not mask.any():
        return []
    g = feats.iloc[mask].copy()
    if min_active_days > 0 and "num_active_days" in g.columns:
        g_active = g[g["num_active_days"] >= min_active_days]
        if not g_active.empty:
            g = g_active
    if g.empty:
        return []
    g["sharpe"] = g["mean_return"] / g["std_return"].replace(0, np.nan)
    valid = g.dropna(subset=["sharpe"]).copy()
    if not valid.empty:
        valid = valid.sort_values("sharpe", ascending=False)
    else:
        valid = g.dropna(subset=["mean_return"]).copy()
        if valid.empty:
            return []
        valid = valid.sort_values("mean_return", ascending=False)
    return list(valid.index[:top_k])


def classify_core_satellite(
    feats: pd.DataFrame,
    labels: np.ndarray,
    cluster_ids: list[int],
    core_std_threshold: float = CORE_STD_THRESHOLD,
    max_dd_exclude: float = MAX_DD_EXCLUDE,
) -> tuple[list[int], list[int]]:
    """
    Partition clusters into Core (low vol, acceptable drawdown) and Satellite (riskier).
    Clusters with median max_drawdown_synth >= max_dd_exclude are excluded entirely.
    """
    core, satellite = [], []
    for c in cluster_ids:
        mask = labels == c
        if not mask.any():
            continue
        med_std = feats.iloc[mask]["std_return"].median()
        med_dd = feats.iloc[mask]["max_drawdown_synth"].median()
        if pd.isna(med_dd) or med_dd >= max_dd_exclude:
            continue
        if not pd.isna(med_std) and med_std <= core_std_threshold:
            core.append(c)
        else:
            satellite.append(c)
    return core, satellite


# ─── Return matrix builder ───────────────────────────────────────────────────

def build_return_matrix_in_window(
    curves: dict[str, pd.Series],
    cluster_to_trader: dict[int, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Aligned daily return matrix for cluster representatives in [start, end]."""
    all_dates: set = set()
    for trader in cluster_to_trader.values():
        if trader not in curves:
            continue
        w = curves[trader]
        window = w[(w.index >= start) & (w.index <= end)].sort_index()
        if len(window) >= 2:
            all_dates.update(window.index.tolist())

    if not all_dates:
        return pd.DataFrame()

    common_index = pd.DatetimeIndex(sorted(all_dates))
    out = pd.DataFrame(index=common_index)
    for cl, trader in cluster_to_trader.items():
        if trader not in curves:
            continue
        eq = curves[trader]
        window = (
            eq[(eq.index >= start) & (eq.index <= end)]
            .reindex(common_index)
            .ffill()
            .bfill()
        )
        if window.isna().all():
            continue
        ret = window.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        out[cl] = ret.reindex(common_index).fillna(0)
    return out.dropna(how="all")


# ─── Layer allocation ────────────────────────────────────────────────────────

def allocate_layers(
    returns_df: pd.DataFrame,
    core_clusters: list[int],
    satellite_clusters: list[int],
    cluster_median_corr: dict[int, float],
    core_ratio: float,
) -> tuple[dict[int, float], dict[int, str]]:
    """
    Risk parity within Core and within Satellite; apply capacity caps;
    combine core_ratio * core + (1-core_ratio) * satellite.
    Returns (cluster_id -> final_weight, cluster_id -> "core"|"satellite").
    """
    layer_tag = {c: "core" for c in core_clusters}
    layer_tag.update({c: "satellite" for c in satellite_clusters})

    def _layer_weights(clusters: list[int]) -> dict[int, float]:
        if not clusters:
            return {}
        in_df = [c for c in clusters if c in returns_df.columns]
        if not in_df:
            return {c: 1.0 / len(clusters) for c in clusters}
        sub = returns_df[in_df].dropna(how="all")
        if sub.empty or len(sub) < 2:
            return {c: 1.0 / len(in_df) for c in in_df}
        sub = sub.replace([np.inf, -np.inf], np.nan).fillna(0)
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            cov = np.nan_to_num(
                np.asarray(sub.cov() * TRADING_DAYS_PER_YEAR),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
        w = equal_risk_contribution_weights(cov)
        w_dict = dict(zip(in_df, w))
        for c in in_df:
            cap = capacity_cap_from_corr(cluster_median_corr.get(c, np.nan))
            w_dict[c] = min(w_dict[c], cap)
        s = sum(w_dict.values())
        if s > 0:
            for c in w_dict:
                w_dict[c] /= s
        return w_dict

    w_core = _layer_weights(core_clusters)
    w_sat = _layer_weights(satellite_clusters)

    r_core = core_ratio if w_core else 0.0
    r_sat = (1.0 - core_ratio) if w_sat else 0.0
    if w_core and not w_sat:
        r_core, r_sat = 1.0, 0.0
    elif w_sat and not w_core:
        r_core, r_sat = 0.0, 1.0

    final: dict[int, float] = {}
    for c, w in w_core.items():
        final[c] = r_core * w
    for c, w in w_sat.items():
        final[c] = final.get(c, 0.0) + r_sat * w
    return final, layer_tag


# ─── Forward return ───────────────────────────────────────────────────────────

def get_forward_returns(
    curves: dict[str, pd.Series],
    trader_weights: dict[str, float],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Portfolio daily return from start to end using trader_weights."""
    all_dates: set = set()
    series = {}
    for trader, w in trader_weights.items():
        if w <= 0 or trader not in curves:
            continue
        eq = curves[trader]
        window = eq[(eq.index >= start) & (eq.index <= end)].sort_index()
        if len(window) < 2:
            continue
        ret = window.pct_change().dropna()
        series[trader] = (ret, w)
        all_dates.update(ret.index.tolist())

    if not all_dates:
        return pd.Series(dtype=float)

    common = pd.DatetimeIndex(sorted(all_dates))
    total = None
    for trader, (ret, w) in series.items():
        r = ret.reindex(common).ffill().bfill().fillna(0)
        total = w * r if total is None else total + w * r
    return total


# ─── Main orchestrator ────────────────────────────────────────────────────────

def rolling_core_satellite_backtest(
    curves_dir: str | Path,
    min_days: int = 60,
    lookback_days: int = 0,
    rebalance_freq: str = "ME",
    core_ratio: float = DEFAULT_CORE_RATIO,
    min_cluster_size: int = 15,
    min_active_days_in_window: int = 15,
    top_traders_per_cluster: int = 1,
    core_std_threshold: float = CORE_STD_THRESHOLD,
    max_dd_exclude: float = MAX_DD_EXCLUDE,
    fee_bps_roundtrip: float = 0.0,
    slippage_bps: float = 0.0,
    max_weight_per_trader: float | None = None,
    output_weights: str | Path = "profiling_step5_rolling_weights.csv",
    output_curve: str | Path = "profiling_step5_rolling_curve.csv",
    output_report: str | Path = "profiling_step5_rolling_report.txt",
    output_fig: str | Path | None = "profiling_step5_rolling_fig.png",
    output_rebalance_stats: str | Path = "profiling_step5_rolling_rebalance_stats.csv",
    output_last_weights: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full rolling core-satellite backtest with no look-ahead.

    At each rebalance:
    1. Compute features from equity curves up to cutoff (PIT).
    2. Standardise and HDBSCAN-cluster.
    3. Classify Core / Satellite.
    4. Risk parity + capacity caps within each layer.
    5. Map cluster weights to trader weights (with optional per-trader cap).
    6. Apply fee + slippage costs based on portfolio turnover.
    7. Forward PnL from next day to next rebalance.

    Returns (weights_df, curve_df).
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    curves_dir = Path(curves_dir)
    if not curves_dir.is_dir():
        raise FileNotFoundError(f"Curves directory not found: {curves_dir}")

    curves = load_all_curves(curves_dir)
    if not curves:
        raise RuntimeError("No equity curves loaded.")

    all_dates_flat = [d for eq in curves.values() for d in eq.index.tolist()]
    min_date = pd.Timestamp(min(all_dates_flat))
    max_date = pd.Timestamp(max(all_dates_flat))

    use_expanding = lookback_days == 0
    if use_expanding:
        first_rebalance = min_date + pd.Timedelta(days=min_days)
        window_start: pd.Timestamp | None = min_date
    else:
        first_rebalance = min_date + pd.Timedelta(days=lookback_days)
        window_start = None

    if first_rebalance >= max_date:
        raise RuntimeError(
            f"Not enough history: first_rebalance={first_rebalance.date()} >= max_date={max_date.date()}."
        )

    rebalance_dates = pd.date_range(start=first_rebalance, end=max_date, freq=rebalance_freq)
    if rebalance_dates.empty:
        raise RuntimeError("No rebalance dates in range.")

    weight_rows: list[dict] = []
    portfolio_returns_list = []
    rebalance_stats_rows: list[dict] = []
    prev_trader_weights: dict[str, float] | None = None

    for i, reb in enumerate(rebalance_dates):
        cutoff = reb
        feats_df = rolling_features_for_cutoff(
            curves, cutoff, lookback_days, min_days,
            window_start=window_start if use_expanding else None,
        )
        if feats_df.empty or len(feats_df) < min_cluster_size * 2:
            continue

        for col in FEATURE_COLUMNS:
            if col not in feats_df.columns:
                feats_df[col] = np.nan
        feats_df = feats_df[FEATURE_COLUMNS].fillna(feats_df.median())
        feats_df = feats_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        for col in feats_df.columns:
            q1, q99 = feats_df[col].quantile(0.01), feats_df[col].quantile(0.99)
            if np.isfinite(q1) and np.isfinite(q99) and q99 > q1:
                feats_df[col] = feats_df[col].clip(lower=q1, upper=q99)

        X = np.nan_to_num(feats_df.values, nan=0.0, posinf=0.0, neginf=0.0)
        if len(X) < min_cluster_size * 2:
            continue

        scaler = StandardScaler()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_std = scaler.fit_transform(X)
        except Exception:
            continue
        X_std = np.nan_to_num(X_std, nan=0.0, posinf=0.0, neginf=0.0)
        if len(X_std) == 0:
            continue

        labels = run_hdbscan(X_std, min_cluster_size=min_cluster_size)
        cluster_ids = [c for c in np.unique(labels) if c != -1]
        if not cluster_ids:
            continue

        core_clusters, satellite_clusters = classify_core_satellite(
            feats_df, labels, cluster_ids, core_std_threshold, max_dd_exclude
        )
        if not core_clusters and not satellite_clusters:
            for c in cluster_ids:
                mask = labels == c
                med_dd = feats_df.iloc[mask]["max_drawdown_synth"].median()
                if pd.isna(med_dd) or med_dd >= max_dd_exclude:
                    continue
                satellite_clusters.append(c)
        if not core_clusters and not satellite_clusters:
            continue

        cluster_to_trader: dict[int, str] = {}
        cluster_median_corr: dict[int, float] = {}
        cluster_to_traders_holdings: dict[int, list[str]] = {}
        for c in cluster_ids:
            t_rep = best_trader_in_cluster(feats_df, labels, c, min_active_days_in_window)
            reps = top_traders_in_cluster(
                feats_df, labels, c, top_traders_per_cluster, min_active_days_in_window
            )
            if t_rep is not None and reps:
                cluster_to_trader[c] = t_rep
                cluster_median_corr[c] = feats_df.iloc[labels == c]["ret_prev_equity_corr"].median()
                cluster_to_traders_holdings[c] = reps

        win_start = window_start if use_expanding else (cutoff - pd.Timedelta(days=lookback_days))
        returns_df = build_return_matrix_in_window(curves, cluster_to_trader, win_start, cutoff)
        if returns_df.empty:
            continue

        final_weights, layer_tag = allocate_layers(
            returns_df, core_clusters, satellite_clusters, cluster_median_corr, core_ratio
        )

        trader_weights: dict[str, float] = {}
        for c, w in final_weights.items():
            if w <= 0 or c not in cluster_to_traders_holdings:
                continue
            reps = cluster_to_traders_holdings[c]
            if not reps:
                continue
            per_trader_weight = w / len(reps)
            for t in reps:
                trader_weights[t] = trader_weights.get(t, 0.0) + per_trader_weight
                weight_rows.append({
                    "rebalance_date": cutoff.strftime("%Y-%m-%d"),
                    "trader": t,
                    "weight": per_trader_weight,
                    "cluster": c,
                    "layer": layer_tag.get(c, ""),
                })

        if max_weight_per_trader is not None and max_weight_per_trader > 0:
            capped: dict[str, float] = {}
            for t, w in trader_weights.items():
                capped[t] = min(w, max_weight_per_trader)
            s = sum(capped.values())
            if s > 0:
                trader_weights = {t: w / s for t, w in capped.items() if w > 0}

        start_fwd = cutoff + pd.Timedelta(days=1)
        end_fwd = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else max_date
        fwd_ret = get_forward_returns(curves, trader_weights, start_fwd, end_fwd)

        turnover = float("nan")
        cost_rate = 0.0
        if not fwd_ret.empty and (fee_bps_roundtrip > 0 or slippage_bps > 0):
            if prev_trader_weights is None:
                turnover = sum(abs(w) for w in trader_weights.values())
            else:
                all_traders = set(prev_trader_weights.keys()) | set(trader_weights.keys())
                turnover = 0.0
                for t in all_traders:
                    w_old = prev_trader_weights.get(t, 0.0)
                    w_new = trader_weights.get(t, 0.0)
                    turnover += abs(w_new - w_old)
            cost_rate = turnover * (fee_bps_roundtrip + slippage_bps) / 10000.0
            if cost_rate > 0 and not fwd_ret.empty:
                first_idx = fwd_ret.index[0]
                fwd_ret.loc[first_idx] = fwd_ret.loc[first_idx] - cost_rate

        if not fwd_ret.empty:
            portfolio_returns_list.append(fwd_ret.rename("portfolio_return"))

        if trader_weights:
            prev_trader_weights = trader_weights

        rebalance_stats_rows.append(
            {
                "rebalance_date": cutoff.strftime("%Y-%m-%d"),
                "n_core_clusters": len(core_clusters),
                "n_satellite_clusters": len(satellite_clusters),
                "n_traders": len(trader_weights),
                "turnover": turnover,
                "fee_bps_roundtrip": fee_bps_roundtrip,
                "slippage_bps": slippage_bps,
                "cost_rate": cost_rate,
            }
        )

    if not weight_rows:
        raise RuntimeError("No weights produced; try smaller --min_days or --min_cluster_size.")

    weights_df = pd.DataFrame(weight_rows)
    weights_df.to_csv(output_weights, index=False)

    if rebalance_stats_rows:
        stats_df = pd.DataFrame(rebalance_stats_rows)
        stats_df.to_csv(output_rebalance_stats, index=False)

    if output_last_weights is not None and not weights_df.empty:
        # Convenience snapshot: trader weights at last rebalance date only.
        last_date = weights_df["rebalance_date"].max()
        last_df = weights_df[weights_df["rebalance_date"] == last_date].copy()
        last_df.to_csv(output_last_weights, index=False)

    curve_df = pd.DataFrame()
    if portfolio_returns_list:
        full_ret = (
            pd.concat(portfolio_returns_list)
            .pipe(lambda s: s[~s.index.duplicated(keep="first")])
            .sort_index()
        )
        equity = (1 + full_ret).cumprod()
        curve_df = pd.DataFrame({"portfolio_return": full_ret, "equity": equity})
        curve_df.index.name = "date"
        curve_df.to_csv(output_curve, index=True)

        # Report
        total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        ann_vol = full_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(full_ret) > 1 else 0
        cum = (1 + full_ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = float(dd.min()) if len(dd) > 0 else 0
        report = [
            "Rolling backtest report (no look-ahead)",
            "=" * 60,
            "",
            "Method: at each rebalance, features + clustering + core/satellite + risk parity "
            "use only data in (window_start, rebalance]. Forward PnL uses only returns "
            "after rebalance date.",
            "",
            f"Window: {'expanding' if use_expanding else str(lookback_days) + ' days lookback'}"
            f"  Rebalance: {rebalance_freq}  Core ratio: {core_ratio:.0%}",
            f"Core = low vol (median std <= {core_std_threshold}), "
            f"Satellite = rest (max_dd < {max_dd_exclude}).",
            "",
            f"Rebalance dates: {len(rebalance_dates)}  Weight rows: {len(weight_rows)}",
            f"Portfolio total return: {total_ret:.2%}  Ann. vol: {ann_vol:.2%}  "
            f"Max drawdown: {max_dd:.2%}",
        ]
        Path(output_report).write_text("\n".join(report), encoding="utf-8")

        # Plot
        if output_fig:
            from src.plotting import plot_rolling_backtest
            plot_rolling_backtest(equity, full_ret, weight_rows, Path(output_fig))

    return weights_df, curve_df
