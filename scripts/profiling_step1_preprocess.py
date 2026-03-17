"""CLI: trader_features.csv -> step1 filtered + standardised features"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.profiling import load_and_filter_features, standardize_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: feature selection & standardisation")
    parser.add_argument("--input", default="data/processed/trader_features.csv")
    parser.add_argument("--output_raw", default="data/processed/profiling_step1_features_raw.csv")
    parser.add_argument("--output_std", default="data/processed/profiling_step1_features_std.csv")
    parser.add_argument("--min_num_days", type=int, default=0)
    parser.add_argument("--min_active_days", type=int, default=20)
    parser.add_argument("--min_active_ratio", type=float, default=0.2)
    parser.add_argument("--max_gap_days", type=float, default=90)
    args = parser.parse_args()

    df = load_and_filter_features(
        args.input,
        min_num_days=args.min_num_days,
        min_active_days=args.min_active_days,
        min_active_ratio=args.min_active_ratio,
        max_gap_days=args.max_gap_days,
    )
    df_std, _ = standardize_features(df)
    df.to_csv(args.output_raw)
    df_std.to_csv(args.output_std)
    print(f"Saved: {args.output_raw}")
    print(f"Saved: {args.output_std}")


if __name__ == "__main__":
    main()
