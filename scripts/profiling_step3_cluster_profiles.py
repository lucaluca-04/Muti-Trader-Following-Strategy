"""CLI: clusters + features -> cluster profiles + best-trader curves"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import build_cluster_profiles, FEATURE_COLUMNS
from src.plotting import plot_best_trader_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: cluster profiles & best-trader curves")
    parser.add_argument("--clusters", default="data/processed/profiling_step3_clusters.csv")
    parser.add_argument("--features", default="data/processed/profiling_step1_features_raw.csv")
    parser.add_argument("--curves_dir", default="data/processed/trader_equity_curves")
    parser.add_argument("--output_csv", default="data/processed/profiling_step3_cluster_profiles.csv")
    parser.add_argument("--output_txt", default="output/reports/profiling_step3_cluster_profiles.txt")
    parser.add_argument("--output_fig", default="output/figures/profiling_step3_best_trader_curves.png")
    parser.add_argument("--output_curves_dir", default="output/figures/profiling_step3_best_trader_curves")
    args = parser.parse_args()

    clusters_df = pd.read_csv(args.clusters)
    features_df = pd.read_csv(args.features)

    profiles_df = build_cluster_profiles(clusters_df, features_df)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(args.output_csv, index=False)

    # Text summary
    lines = ["Cluster profiles", "=" * 60]
    for _, row in profiles_df.sort_values("cluster").iterrows():
        cl, n, st = row["cluster"], int(row["count"]), row["strategy_type"]
        lines.append(f"\nCluster {cl}  (n={n})  strategy_type: {st}")
        if cl != -1:
            lines.append(
                f"  mean_return={row.get('median_mean_return', np.nan):.4f}  "
                f"std_return={row.get('median_std_return', np.nan):.4f}  "
                f"max_drawdown_synth={row.get('median_max_drawdown_synth', np.nan):.4f}"
            )
    Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_txt).write_text("\n".join(lines), encoding="utf-8")

    # Best-trader curves figure
    curves_dir = Path(args.curves_dir)
    if curves_dir.is_dir():
        feats_idx = features_df.set_index("trader") if "trader" in features_df.columns else features_df
        merged = clusters_df.set_index("trader", drop=False).join(
            feats_idx[FEATURE_COLUMNS], how="inner"
        )
        plot_best_trader_curves(
            merged, curves_dir, args.output_fig,
            output_dir=Path(args.output_curves_dir),
        )

    print(f"Saved: {args.output_csv}")
    print(f"Saved: {args.output_txt}")


if __name__ == "__main__":
    main()
