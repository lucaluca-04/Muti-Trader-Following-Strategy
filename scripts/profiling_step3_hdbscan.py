"""CLI: UMAP embedding -> HDBSCAN cluster labels"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import run_hdbscan


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: HDBSCAN clustering")
    parser.add_argument("--input", default="data/processed/profiling_step2_embedding.csv")
    parser.add_argument("--output", default="data/processed/profiling_step3_clusters.csv")
    parser.add_argument("--min_cluster_size", type=int, default=15)
    parser.add_argument("--min_samples", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col="trader")
    labels = run_hdbscan(df.values, min_cluster_size=args.min_cluster_size,
                         min_samples=args.min_samples)
    out = pd.DataFrame({"trader": df.index.values, "cluster": labels})
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}  ({len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
          f"{(labels == -1).sum()} noise)")


if __name__ == "__main__":
    main()
