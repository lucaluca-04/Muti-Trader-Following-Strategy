from __future__ import annotations

"""
Profiling utilities (Step 1–4) shared across scripts.

Combines and exposes:
- Step 1: feature selection & standardisation
- Step 2: UMAP embedding
- Step 3: HDBSCAN clustering + cluster profile building
- Step 4: cluster allocation (risk parity + Sharpe tilt + capacity caps) & style drift
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import umap as _umap  # type: ignore
except ImportError:
    _umap = None

try:
    import hdbscan as _hdbscan  # type: ignore
except ImportError:
    _hdbscan = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ─── Shared feature column list ──────────────────────────────────────────────

FEATURE_COLUMNS: List[str] = [
    "mean_return",
    "std_return",
    "max_drawdown_synth",
    "tail_avg_5pct",
    "p5_return",
    "ret_prev_equity_corr",
    "avg_equity",
    "max_net_deposit",
    "mean_return_big_equity",
    "mean_return_small_equity",
    "acf1_return",
    "acf1_abs_return",
]

TRADING_DAYS_PER_YEAR = 252


# ─── Step 1: Feature selection & standardisation ──────────────────────────────

def load_and_filter_features(
    csv_path: str | Path,
    min_num_days: int = 0,
    min_active_days: int = 0,
    min_active_ratio: float = 0.0,
    max_gap_days: float = 0,
) -> pd.DataFrame:
    """
    Load trader_features.csv, apply quality filters and return a DataFrame
    indexed by trader containing FEATURE_COLUMNS as floats.
    """
    df = pd.read_csv(csv_path)
    df = df[df["is_kept"] == 1].copy()

    if min_num_days > 0 and "num_return_days" in df.columns:
        df["num_return_days"] = pd.to_numeric(df["num_return_days"], errors="coerce")
        df = df[df["num_return_days"] >= min_num_days].copy()

    if min_active_days > 0 and "num_active_days" in df.columns:
        df["num_active_days"] = pd.to_numeric(df["num_active_days"], errors="coerce")
        df = df[df["num_active_days"] >= min_active_days].copy()

    if min_active_ratio > 0 and "active_day_ratio" in df.columns:
        df["active_day_ratio"] = pd.to_numeric(df["active_day_ratio"], errors="coerce")
        df = df[df["active_day_ratio"] >= min_active_ratio].copy()

    if max_gap_days > 0 and "max_gap_calendar_days" in df.columns:
        df["max_gap_calendar_days"] = pd.to_numeric(df["max_gap_calendar_days"], errors="coerce")
        df = df[df["max_gap_calendar_days"] <= max_gap_days].copy()

    if "trader" in df.columns:
        df = df.set_index("trader")

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Required feature column '{col}' not found in {csv_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def standardize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Z-score standardise FEATURE_COLUMNS. Missing values are filled with column medians.
    Returns (df with added *_z columns, raw standardised matrix X).
    """
    df_features = df[FEATURE_COLUMNS].copy().fillna(df[FEATURE_COLUMNS].median())
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features.values)
    df_std = df.copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        df_std[col + "_z"] = X[:, i]
    return df_std, X


# ─── Step 2: UMAP ─────────────────────────────────────────────────────────────

def run_umap_on_X(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """UMAP on trader×feature matrix X; returns embedding of shape (n_traders, n_components)."""
    if _umap is None:
        raise ImportError("umap-learn is not installed. Please run: pip install umap-learn")
    model = _umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    return model.fit_transform(X)


def denoise_correlation(X: np.ndarray, eigenvalue_floor_quantile: float = 0.2) -> np.ndarray:
    """Simple constant-residual eigenvalue denoising on the feature correlation matrix."""
    C = np.corrcoef(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    threshold = np.quantile(eigvals, eigenvalue_floor_quantile)
    floored = np.where(eigvals < threshold, threshold, eigvals)
    C_denoised = (eigvecs * floored) @ eigvecs.T
    d = np.sqrt(np.diag(C_denoised))
    C_denoised = C_denoised / np.outer(d, d)
    return C_denoised


# ─── Step 3: HDBSCAN clustering ───────────────────────────────────────────────

def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: int | None = None,
) -> np.ndarray:
    """HDBSCAN in UMAP space. Returns cluster labels (-1 = noise)."""
    if _hdbscan is None:
        raise ImportError("hdbscan is not installed. Please run: pip install hdbscan")
    model = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    return model.fit_predict(embedding)


# ─── Step 3: Cluster profiles ─────────────────────────────────────────────────

def assign_strategy_type(row: pd.Series, global_median: pd.Series) -> str:
    """Assign a short English strategy-type label from cluster profile vs global median."""
    tags = []
    mr = row.get("mean_return", np.nan)
    sr = row.get("std_return", np.nan)
    dd = row.get("max_drawdown_synth", np.nan)
    corr_eq = row.get("ret_prev_equity_corr", np.nan)
    ae = row.get("avg_equity", np.nan)
    acf1 = row.get("acf1_return", np.nan)
    acf1_abs = row.get("acf1_abs_return", np.nan)
    g_sr = global_median.get("std_return", 0) or 1e-8
    g_mr = global_median.get("mean_return", 0)
    g_ae = global_median.get("avg_equity", 0) or 1
    g_dd = global_median.get("max_drawdown_synth", 0) or 1e-8

    if not (np.isnan(mr) or np.isnan(sr)):
        if mr > g_mr and sr > g_sr:
            tags.append("High risk / High return")
        elif sr < g_sr * 0.7:
            tags.append("Low volatility")
        elif mr > g_mr:
            tags.append("Higher mean return")
        elif sr > g_sr * 1.3:
            tags.append("High volatility")

    if not np.isnan(dd) and g_dd > 0 and dd > g_dd * 1.2:
        tags.append("Higher tail risk")

    if not np.isnan(corr_eq):
        if corr_eq > 0.15:
            tags.append("Capacity sensitive (AUM-dependent)")
        elif corr_eq < -0.15:
            tags.append("Negative AUM sensitivity")

    if not np.isnan(ae) and ae > g_ae * 2:
        tags.append("Large size")
    elif not np.isnan(ae) and ae < g_ae * 0.5 and g_ae > 0:
        tags.append("Small size")

    if not np.isnan(acf1) and not np.isnan(acf1_abs):
        if acf1 > 0.1:
            tags.append("Momentum / Trend")
        elif acf1 < -0.1:
            tags.append("Mean-reversion")
        if acf1_abs > 0.2:
            tags.append("Vol clustering")

    return " | ".join(tags[:4]) if tags else "Mixed / Unclassified"


def best_trader_per_cluster(df: pd.DataFrame, cluster: int) -> str | None:
    """Pick the trader in the cluster with highest ex-ante Sharpe (mean_return / std_return)."""
    g = df[df["cluster"] == cluster].copy()
    if g.empty:
        return None
    g["std_return"] = pd.to_numeric(g["std_return"], errors="coerce").replace(0, np.nan)
    g["sharpe"] = g["mean_return"] / g["std_return"]
    valid = g.dropna(subset=["sharpe"])
    if not valid.empty:
        return valid.loc[valid["sharpe"].idxmax(), "trader"]
    return g.loc[g["mean_return"].idxmax(), "trader"]


def build_cluster_profiles(
    clusters_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge cluster labels with raw features; compute per-cluster mean/median stats
    and assign strategy_type labels.
    Returns one row per cluster.
    """
    clusters = clusters_df.set_index("trader", drop=False)
    feats = features_df.set_index("trader") if "trader" in features_df.columns else features_df
    for col in FEATURE_COLUMNS:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")

    common = clusters.index.intersection(feats.index)
    df = clusters.loc[common].copy()
    df = df.join(feats[FEATURE_COLUMNS], how="inner")
    global_median = feats.loc[common][FEATURE_COLUMNS].median()

    profile_rows = []
    for cl, g in df.groupby("cluster"):
        row: dict = {"cluster": cl, "count": len(g)}
        for col in FEATURE_COLUMNS:
            row[f"mean_{col}"] = g[col].mean()
            row[f"median_{col}"] = g[col].median()
        row["strategy_type"] = assign_strategy_type(g[FEATURE_COLUMNS].median(), global_median)
        profile_rows.append(row)

    out = pd.DataFrame(profile_rows)
    base = ["cluster", "count", "strategy_type"]
    rest = sorted([c for c in out.columns if c not in base])
    return out[base + rest]


# ─── Step 4: Cluster allocation ───────────────────────────────────────────────

def build_aligned_return_matrix(
    curves_dir: Path,
    cluster_to_trader: dict[int, str],
) -> tuple[pd.DataFrame, dict]:
    """
    Build a matrix of daily returns aligned on a common date index.
    Columns = cluster ids. Returns (returns_df, {cluster: raw_series}).
    """
    from src.dataio import load_equity_curve  # avoid circular at module level

    all_dates: set = set()
    series_by_cluster: dict[int, pd.Series] = {}

    for cluster, trader in cluster_to_trader.items():
        curve = load_equity_curve(curves_dir, trader)
        if curve is None or len(curve) < 2:
            continue
        curve = curve.set_index("date")
        ret = curve["equity_twr"].pct_change().dropna()
        ret = ret[ret.notna()]
        if ret.empty:
            continue
        series_by_cluster[cluster] = ret
        all_dates.update(ret.index.tolist())

    if not all_dates:
        return pd.DataFrame(), {}

    common_index = pd.DatetimeIndex(sorted(all_dates))
    aligned = {}
    for cluster, trader in cluster_to_trader.items():
        curve = load_equity_curve(curves_dir, trader)
        if curve is None or len(curve) < 2:
            continue
        curve = curve.set_index("date").reindex(common_index).ffill().bfill()
        if curve["equity_twr"].isna().all():
            continue
        ret = curve["equity_twr"].pct_change().dropna()
        ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0)
        aligned[cluster] = ret

    if not aligned:
        return pd.DataFrame(), {}

    returns_df = pd.DataFrame(aligned).dropna(how="all")
    return returns_df, series_by_cluster


def compute_cluster_metrics(
    returns_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Annualised return, vol, Sharpe, max drawdown per column (cluster)."""
    if len(returns_df) == 0:
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()
    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)
    mean_r = returns_df.mean()
    vol = returns_df.std()
    ann_ret = mean_r * TRADING_DAYS_PER_YEAR
    ann_vol = vol * ann_factor
    sharpe = (ann_ret / ann_vol.replace(0, np.nan)).fillna(0)
    cum = (1 + returns_df).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    return ann_ret, ann_vol, sharpe, max_dd


def equal_risk_contribution_weights(cov: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Iterative risk-parity (equal risk contribution) given annualised covariance matrix."""
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    vol = np.sqrt(np.diag(cov))
    vol = np.where(vol > 1e-12, vol, 1e-12)
    w = (1.0 / vol) / (1.0 / vol).sum()
    for _ in range(max_iter):
        marginal = cov @ w
        w_new = 1.0 / np.where(marginal > 1e-12, marginal, 1e-12)
        w_new /= w_new.sum()
        if np.allclose(w, w_new, atol=1e-8):
            break
        w = w_new
    return w


def capacity_cap_from_corr(corr_val: float) -> float:
    """More negative ret_prev_equity_corr => stronger capacity constraint => lower weight cap."""
    if pd.isna(corr_val):
        return 0.25
    if corr_val <= -0.7:
        return 0.10
    if corr_val <= -0.4:
        return 0.15
    if corr_val <= -0.15:
        return 0.20
    return 0.25


# ─── Step 4: Style drift ──────────────────────────────────────────────────────

def compute_drift_score(cluster_series: np.ndarray) -> float:
    """
    Normalised cluster-label entropy for a trader over time windows.
    0 = always same cluster (stable), 1 = fully scattered or all-noise (drifting).
    """
    if cluster_series.size == 0:
        return float("nan")
    valid = cluster_series[cluster_series != -1]
    if valid.size == 0:
        return 1.0
    _, counts = np.unique(valid, return_counts=True)
    p = counts / counts.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    max_entropy = np.log(len(counts))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)
