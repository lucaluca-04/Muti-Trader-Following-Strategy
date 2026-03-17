"""CLI: daily returns -> per-trader equity curve CSVs"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataio import export_equity_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-trader TWR equity curves")
    parser.add_argument("--daily_ret_csv", default="data/processed/trader_daily_returns.csv")
    parser.add_argument("--output_dir", default="data/processed/trader_equity_curves")
    args = parser.parse_args()

    n = export_equity_curves(args.daily_ret_csv, args.output_dir)
    print(f"Wrote {n} curves to {args.output_dir}/")


if __name__ == "__main__":
    main()
