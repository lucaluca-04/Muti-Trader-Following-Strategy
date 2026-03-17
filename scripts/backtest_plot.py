"""CLI: portfolio_returns CSV -> equity + drawdown plot"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.plotting import plot_equity_drawdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot equity curve and drawdown from returns CSV")
    parser.add_argument("--input", default="data/processed/backtest_portfolio_returns.csv")
    parser.add_argument("--output", default="output/figures/backtest_summary.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    plot_equity_drawdown(df["portfolio_return"], args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
