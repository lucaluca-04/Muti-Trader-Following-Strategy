"""CLI: clusters + features + profiles -> allocation weights"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import (
    FEATURE_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    best_trader_per_cluster,
    build_aligned_return_matrix,
    compute_cluster_metrics,
    equal_risk_contribution_weights,
    capacity_cap_from_corr,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: cluster allocation")
    parser.add_argument("--clusters", default="data/processed/profiling_step3_clusters.csv")
    parser.add_argument("--features", default="data/processed/profiling_step1_features_raw.csv")
    parser.add_argument("--profiles", default="data/processed/profiling_step3_cluster_profiles.csv")
    parser.add_argument("--curves_dir", default="data/processed/trader_equity_curves")
    parser.add_argument("--max_dd_exclude", type=float, default=0.95)
    parser.add_argument("--sharpe_tilt", type=float, default=1.0)
    parser.add_argument("--output_weights", default="data/processed/profiling_step4_allocation_weights.csv")
    parser.add_argument("--output_report", default="output/reports/profiling_step4_allocation_report.txt")
    args = parser.parse_args()

    curves_dir = Path(args.curves_dir)
    clusters_df = pd.read_csv(args.clusters)
    features_df = pd.read_csv(args.features)
    profiles_df = pd.read_csv(args.profiles)

    clusters = clusters_df.set_index("trader", drop=False)
    feats = features_df.set_index("trader")
    for col in FEATURE_COLUMNS:
        feats[col] = pd.to_numeric(feats[col], errors="coerce")
    common = clusters.index.intersection(feats.index)
    df = clusters.loc[common].copy().join(feats[FEATURE_COLUMNS], how="inner")

    profile_by_cluster = profiles_df.set_index("cluster")
    clusters_sorted = sorted([c for c in df["cluster"].unique() if c != -1 and isinstance(c, (int, np.integer))])
    cluster_to_trader = {cl: best_trader_per_cluster(df, cl) for cl in clusters_sorted}
    cluster_to_trader = {cl: t for cl, t in cluster_to_trader.items() if t is not None}

    returns_df, _ = build_aligned_return_matrix(curves_dir, cluster_to_trader)
    if returns_df.empty:
        raise RuntimeError("No return data for any cluster.")

    ann_ret, ann_vol, sharpe, max_dd_emp = compute_cluster_metrics(returns_df)

    eligible = [
        cl for cl in returns_df.columns
        if cl not in profile_by_cluster.index
        or pd.isna(profile_by_cluster.loc[cl].get("median_max_drawdown_synth", np.nan))
        or profile_by_cluster.loc[cl].get("median_max_drawdown_synth") < args.max_dd_exclude
    ]
    if not eligible:
        raise RuntimeError("No cluster passed the max_drawdown exclusion.")

    returns_sub = returns_df[eligible].dropna(how="all")
    cov = np.nan_to_num(np.asarray(returns_sub.cov() * TRADING_DAYS_PER_YEAR), nan=0.0)
    w_rp = pd.Series(equal_risk_contribution_weights(cov), index=eligible)

    alpha = args.sharpe_tilt
    quality = np.where(np.isfinite(sharpe[eligible].values), np.maximum(sharpe[eligible].values, 0.0) ** alpha, 0.0)
    tilt = quality / quality.sum() * len(quality) if quality.sum() > 0 else np.ones_like(quality)
    w_tilted = w_rp * tilt
    w_tilted = (w_tilted / w_tilted.sum()).where(np.isfinite(w_rp * tilt), w_rp)

    caps = {
        cl: capacity_cap_from_corr(
            profile_by_cluster.loc[cl].get("median_ret_prev_equity_corr", np.nan)
            if cl in profile_by_cluster.index else np.nan
        )
        for cl in eligible
    }
    w_capped = w_tilted.copy()
    for cl in eligible:
        w_capped[cl] = min(w_capped[cl], caps[cl])
    if w_capped.sum() > 0:
        w_capped /= w_capped.sum()

    rows = [
        {
            "cluster": cl,
            "trader": cluster_to_trader[cl],
            "weight": w_capped[cl],
            "weight_risk_parity": w_rp[cl],
            "ann_ret": ann_ret.get(cl, np.nan),
            "ann_vol": ann_vol.get(cl, np.nan),
            "sharpe": sharpe.get(cl, np.nan),
            "max_dd_emp": max_dd_emp.get(cl, np.nan),
            "capacity_cap": caps[cl],
        }
        for cl in eligible
    ]
    out_df = pd.DataFrame(rows)
    Path(args.output_weights).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_weights, index=False)

    report = [
        "Cluster-based allocation report",
        "=" * 60,
        f"Eligible clusters: {eligible}",
        f"Max DD exclude: {args.max_dd_exclude}  Sharpe tilt: {args.sharpe_tilt}",
        "",
    ]
    for _, r in out_df.iterrows():
        report.append(
            f"  Cluster {int(r['cluster']):2d}  weight={r['weight']:.2%}  "
            f"cap={r['capacity_cap']:.2%}  Sharpe={r['sharpe']:.3f}  "
            f"ann_vol={r['ann_vol']:.2%}  ann_ret={r['ann_ret']:.2%}"
        )
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text("\n".join(report), encoding="utf-8")
    print(f"Saved: {args.output_weights}")
    print(f"Saved: {args.output_report}")


if __name__ == "__main__":
    main()
