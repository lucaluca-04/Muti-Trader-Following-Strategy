"""CLI: daily returns + raw CSV -> trader_features.csv"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import build_trader_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-trader features")
    parser.add_argument("--raw_csv", default="data/raw/笔试题数据包.csv")
    parser.add_argument("--daily_ret_csv", default="data/processed/trader_daily_returns.csv")
    parser.add_argument("--perf_csv", default="data/processed/trader_performance_raw.csv")
    parser.add_argument("--output", default="data/processed/trader_features.csv")
    args = parser.parse_args()

    build_trader_features(
        raw_csv=args.raw_csv,
        daily_ret_csv=args.daily_ret_csv,
        perf_csv=args.perf_csv,
        output_features_csv=args.output,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
