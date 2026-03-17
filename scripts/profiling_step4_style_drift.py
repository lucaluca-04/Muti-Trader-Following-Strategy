"""CLI: cluster history -> style drift scores"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import compute_drift_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: style drift analysis")
    parser.add_argument("--input", default="data/processed/profiling_style_history.csv")
    parser.add_argument("--fallback_snapshot", default="data/processed/profiling_step3_clusters.csv")
    parser.add_argument("--output", default="data/processed/profiling_step4_style_drift.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file() and Path(args.fallback_snapshot).is_file():
        snap = pd.read_csv(args.fallback_snapshot)
        snap["window_id"] = 0
        df = snap[["trader", "window_id", "cluster"]].copy()
    else:
        df = pd.read_csv(input_path)

    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
    df = df.sort_values(["trader", "window_id"])

    rows = []
    for trader, g in df.groupby("trader"):
        clusters = g["cluster"].dropna().astype(int).to_numpy()
        rows.append({
            "trader": trader,
            "style_drift_score": compute_drift_score(clusters),
            "num_windows": len(g),
            "num_valid_windows": int((clusters != -1).sum()),
        })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
