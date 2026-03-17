from __future__ import annotations

"""
High-level pipeline entrypoint.

Reads config.yaml and runs the full pipeline:
  1. Raw CSV -> daily returns + performance table
  2. Feature engineering -> trader_features.csv
  3. Export per-trader equity curves
  4. Profiling (step 1-4): filter, UMAP, HDBSCAN, cluster profiles, allocation
  5. Rolling backtest (step 5)
"""

from pathlib import Path

import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    root = Path(__file__).parent
    config = load_config(root / "config.yaml")

    raw_dir = root / config["data"]["raw_dir"]
    processed_dir = root / config["data"]["processed_dir"]
    figures_dir = root / config["output"]["figures_dir"]
    reports_dir = root / config["output"]["reports_dir"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / "笔试题数据包.csv"
    daily_ret_csv = processed_dir / "trader_daily_returns.csv"
    perf_csv = processed_dir / "trader_performance_raw.csv"
    features_csv = processed_dir / "trader_features.csv"
    curves_dir = processed_dir / "trader_equity_curves"

    # Step 1: raw -> daily returns + performance
    print("[1/5] Computing trader performance...")
    from src.performance import compute_trader_performance
    compute_trader_performance(
        input_csv=raw_csv,
        output_daily_returns_csv=daily_ret_csv,
        output_summary_csv=perf_csv,
    )

    # Step 2: feature engineering
    print("[2/5] Building trader features...")
    from src.features import build_trader_features
    build_trader_features(
        raw_csv=raw_csv,
        daily_ret_csv=daily_ret_csv,
        perf_csv=perf_csv,
        output_features_csv=features_csv,
    )

    # Step 3: export equity curves
    print("[3/5] Exporting equity curves...")
    from src.dataio import export_equity_curves
    n = export_equity_curves(daily_ret_csv, curves_dir)
    print(f"       {n} curves written to {curves_dir}")

    # Step 4: profiling pipeline (step 1-4)
    print("[4/5] Running profiling pipeline (UMAP + HDBSCAN + allocation)...")
    from src.profiling import (
        load_and_filter_features,
        standardize_features,
        run_umap_on_X,
        run_hdbscan,
        build_cluster_profiles,
        best_trader_per_cluster,
        build_aligned_return_matrix,
        compute_cluster_metrics,
        equal_risk_contribution_weights,
        capacity_cap_from_corr,
        FEATURE_COLUMNS,
    )
    import numpy as np
    import pandas as pd

    df_raw = load_and_filter_features(
        features_csv, min_active_days=20, min_active_ratio=0.2, max_gap_days=90
    )
    df_std, X = standardize_features(df_raw)
    df_raw.to_csv(processed_dir / "profiling_step1_features_raw.csv")
    df_std.to_csv(processed_dir / "profiling_step1_features_std.csv")

    FEATURE_COLUMNS_Z = [c + "_z" for c in FEATURE_COLUMNS]
    embedding = run_umap_on_X(df_std[FEATURE_COLUMNS_Z].values)
    emb_df = pd.DataFrame(
        embedding, index=df_std.index,
        columns=[f"umap{i+1}" for i in range(embedding.shape[1])]
    )
    emb_df.to_csv(processed_dir / "profiling_step2_embedding.csv")

    labels = run_hdbscan(embedding, min_cluster_size=config["backtest"]["min_cluster_size"])
    clusters_df = pd.DataFrame({"trader": df_std.index.values, "cluster": labels})
    clusters_df.to_csv(processed_dir / "profiling_step3_clusters.csv", index=False)

    profiles_df = build_cluster_profiles(clusters_df, df_raw.reset_index())
    profiles_df.to_csv(processed_dir / "profiling_step3_cluster_profiles.csv", index=False)

    from src.plotting import plot_best_trader_curves
    merged = clusters_df.set_index("trader", drop=False).join(
        df_raw[FEATURE_COLUMNS], how="inner"
    )
    plot_best_trader_curves(
        merged,
        curves_dir,
        figures_dir / "profiling_step3_best_trader_curves.png",
        output_dir=figures_dir / "profiling_step3_best_trader_curves",
    )

    # Step 5: rolling backtest
    print("[5/5] Running rolling backtest...")
    from src.backtest import rolling_core_satellite_backtest
    rolling_core_satellite_backtest(
        curves_dir=curves_dir,
        min_days=config["backtest"]["min_days"],
        lookback_days=config["backtest"]["lookback_days"],
        rebalance_freq=config["backtest"]["rebalance_freq"],
        core_ratio=config["backtest"]["core_ratio"],
        min_cluster_size=config["backtest"]["min_cluster_size"],
        top_traders_per_cluster=int(config["backtest"]["top_traders_per_cluster"]),
        fee_bps_roundtrip=(
            float(config["fees"]["taker_fee_bps"])
            + float(config["fees"]["maker_fee_bps"])
        ),
        slippage_bps=float(config["slippage"]["fixed_bps"]),
        max_weight_per_trader=float(config["capacity"]["max_weight_per_trader"]),
        output_weights=processed_dir / "profiling_step5_rolling_weights.csv",
        output_curve=processed_dir / "profiling_step5_rolling_curve.csv",
        output_report=reports_dir / "profiling_step5_rolling_report.txt",
        output_fig=figures_dir / "profiling_step5_rolling_fig.png",
        output_rebalance_stats=processed_dir / "profiling_step5_rolling_rebalance_stats.csv",
        output_last_weights=processed_dir / "profiling_step5_rolling_last_weights.csv",
    )

    print("\nDone. Outputs:")
    print(f"  Data:    {processed_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Reports: {reports_dir}")


if __name__ == "__main__":
    main()
