"""CLI: step1 standardised features -> UMAP embedding"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import run_umap_on_X, FEATURE_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: UMAP embedding")
    parser.add_argument("--input", default="data/processed/profiling_step1_features_std.csv")
    parser.add_argument("--output", default="data/processed/profiling_step2_embedding.csv")
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col="trader")
    z_cols = [c + "_z" for c in FEATURE_COLUMNS]
    X = df[z_cols].values

    embedding = run_umap_on_X(
        X, n_components=args.n_components,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
    )
    out = pd.DataFrame(
        embedding, index=df.index,
        columns=[f"umap{i+1}" for i in range(embedding.shape[1])],
    )
    out.to_csv(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
