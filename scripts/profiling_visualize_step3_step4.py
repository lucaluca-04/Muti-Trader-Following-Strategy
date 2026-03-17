"""CLI: step3 clusters + step4 drift -> summary figure"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.plotting import plot_cluster_drift_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise step3 clusters and step4 style drift")
    parser.add_argument("--step3", default="data/processed/profiling_step3_clusters.csv")
    parser.add_argument("--step4", default="data/processed/profiling_step4_style_drift.csv")
    parser.add_argument("--output", default="output/figures/profiling_step3_step4_summary.png")
    args = parser.parse_args()

    clusters_df = pd.read_csv(args.step3)
    drift_df = pd.read_csv(args.step4)
    plot_cluster_drift_summary(clusters_df, drift_df, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
