"""CLI: equity curves -> rolling core-satellite backtest"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.backtest import rolling_core_satellite_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: rolling core-satellite backtest (no look-ahead)")
    parser.add_argument("--curves_dir", default="data/processed/trader_equity_curves")
    parser.add_argument("--lookback", type=int, default=0,
                        help="Lookback days; 0 = expanding window (default)")
    parser.add_argument("--min_days", type=int, default=60)
    parser.add_argument("--rebalance_freq", choices=["ME", "W", "2ME"], default="ME")
    parser.add_argument("--core_ratio", type=float, default=0.80)
    parser.add_argument("--min_cluster_size", type=int, default=15)
    parser.add_argument("--min_active_days_in_window", type=int, default=15)
    parser.add_argument("--output_weights", default="data/processed/profiling_step5_rolling_weights.csv")
    parser.add_argument("--output_curve", default="data/processed/profiling_step5_rolling_curve.csv")
    parser.add_argument("--output_report", default="output/reports/profiling_step5_rolling_report.txt")
    parser.add_argument("--output_fig", default="output/figures/profiling_step5_rolling_fig.png")
    args = parser.parse_args()

    rolling_core_satellite_backtest(
        curves_dir=args.curves_dir,
        min_days=args.min_days,
        lookback_days=args.lookback,
        rebalance_freq=args.rebalance_freq,
        core_ratio=args.core_ratio,
        min_cluster_size=args.min_cluster_size,
        min_active_days_in_window=args.min_active_days_in_window,
        output_weights=args.output_weights,
        output_curve=args.output_curve,
        output_report=args.output_report,
        output_fig=args.output_fig,
    )


if __name__ == "__main__":
    main()
