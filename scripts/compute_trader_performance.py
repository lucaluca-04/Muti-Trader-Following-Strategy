"""CLI: raw CSV -> trader_daily_returns.csv + trader_performance_raw.csv"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.performance import compute_trader_performance


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-trader daily returns and performance")
    parser.add_argument("--input", default="data/raw/笔试题数据包.csv")
    parser.add_argument("--output_daily", default="data/processed/trader_daily_returns.csv")
    parser.add_argument("--output_summary", default="data/processed/trader_performance_raw.csv")
    args = parser.parse_args()

    compute_trader_performance(
        input_csv=args.input,
        output_daily_returns_csv=args.output_daily,
        output_summary_csv=args.output_summary,
    )
    print(f"Saved: {args.output_daily}")
    print(f"Saved: {args.output_summary}")


if __name__ == "__main__":
    main()
