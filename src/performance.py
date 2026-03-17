from __future__ import annotations

"""
Performance-related utilities shared across scripts.

This module centralizes:
- safe float parsing
- IRR (money-weighted return) computation
- core logic currently used in compute_trader_performance.py
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_irr(cashflows: Iterable[float], max_iter: int = 100, tol: float = 1e-6) -> float:
    """
    Simple IRR via bisection on equally spaced periods.
    Ported from compute_trader_performance.main for reuse.
    """
    cashflows = list(cashflows)
    if not cashflows:
        return float("nan")

    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return float("nan")

    def npv(rate: float) -> float:
        if rate <= -1.0:
            return float("inf")
        total = 0.0
        denom = 1.0
        for cf in cashflows:
            total += cf / denom
            denom *= 1.0 + rate
            if denom == 0.0:
                denom = float("inf")
        return total

    low = -0.9999
    high = 10.0
    npv_low = npv(low)
    npv_high = npv(high)

    expand_factor = 2.0
    attempts = 0
    while npv_low * npv_high > 0 and attempts < 10:
        high *= expand_factor
        npv_high = npv(high)
        attempts += 1

    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = npv(mid)
        if abs(npv_mid) < tol:
            return mid
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return (low + high) / 2.0


def compute_trader_performance(
    input_csv: str | Path,
    output_daily_returns_csv: str | Path,
    output_summary_csv: str | Path,
) -> None:
    """
    Library-style wrapper for the logic in compute_trader_performance.py:main.
    """
    input_csv = Path(input_csv)
    output_daily_returns_csv = Path(output_daily_returns_csv)
    output_summary_csv = Path(output_summary_csv)

    by_trader: dict[str, list[dict]] = defaultdict(list)

    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trader = row.get("trader") or row.get("user")
            if not trader:
                continue
            date = row.get("date")
            if not date:
                continue
            equity = safe_float(row.get("equity"))
            net_deposit = safe_float(row.get("net_deposit"))
            pnl = safe_float(row.get("pnl"))
            by_trader[trader].append(
                {
                    "date": date,
                    "equity": equity,
                    "net_deposit": net_deposit,
                    "pnl": pnl,
                }
            )

    daily_returns_rows: list[dict] = []
    summary_rows: list[dict] = []

    for trader, rows in by_trader.items():
        rows.sort(key=lambda r: r["date"])

        returns: list[float] = []
        equity_curve: list[float] = []

        prev_equity = None
        prev_net_dep = None
        prev_pnl = None

        cashflows: list[float] = []
        irr_prev_net_dep = 0.0

        for idx, r in enumerate(rows):
            date = r["date"]
            equity = r["equity"]
            net_dep = r["net_deposit"]
            pnl = r["pnl"]

            delta_dep = net_dep - irr_prev_net_dep
            cf = -delta_dep
            if abs(cf) > 1e-8:
                cashflows.append(cf)
            irr_prev_net_dep = net_dep

            if prev_equity is None:
                prev_equity = equity
                prev_net_dep = net_dep
                prev_pnl = pnl
                equity_curve.append(equity)
                daily_returns_rows.append({"trader": trader, "date": date, "daily_return": ""})
                continue

            flow = net_dep - prev_net_dep
            period_pnl = pnl - prev_pnl

            if prev_equity != 0 and not math.isnan(prev_equity):
                daily_return = period_pnl / prev_equity
                if math.isfinite(daily_return) and (daily_return > 1.0 or daily_return < -1.0):
                    daily_return = 1.0 if daily_return > 1.0 else -1.0
            else:
                daily_return = float("nan")

            if not math.isnan(daily_return) and math.isfinite(daily_return):
                returns.append(daily_return)
                daily_returns_rows.append(
                    {"trader": trader, "date": date, "daily_return": f"{daily_return:.10f}"}
                )
            else:
                daily_returns_rows.append({"trader": trader, "date": date, "daily_return": ""})

            if not equity_curve:
                equity_curve.append(prev_equity)
            last_eq = equity_curve[-1]
            synthetic_eq = last_eq * (1.0 + (daily_return if not math.isnan(daily_return) else 0.0))
            equity_curve.append(synthetic_eq)

            prev_equity = equity
            prev_net_dep = net_dep
            prev_pnl = pnl

        if rows:
            last_equity = rows[-1]["equity"]
            if not math.isnan(last_equity):
                cashflows.append(last_equity)

        n_returns = len(returns)
        total_twr = float("nan")
        annual_twr = float("nan")
        mean_ret = float("nan")
        std_ret = float("nan")
        annual_vol = float("nan")
        sharpe = float("nan")
        max_dd = float("nan")
        calmar = float("nan")
        irr = float("nan")

        if n_returns > 0:
            prod = 1.0
            for r in returns:
                prod *= 1.0 + r
            total_twr = prod - 1.0

            if 1.0 + total_twr > 0 and n_returns > 0:
                try:
                    annual_twr = (1.0 + total_twr) ** (252.0 / n_returns) - 1.0
                except OverflowError:
                    annual_twr = float("nan")

            mean_ret = sum(returns) / n_returns
            if n_returns > 1:
                var = sum((r - mean_ret) ** 2 for r in returns) / (n_returns - 1)
                std_ret = math.sqrt(var)
            else:
                std_ret = 0.0

            annual_vol = std_ret * math.sqrt(252.0)

            if std_ret > 0:
                sharpe = mean_ret / std_ret * math.sqrt(252.0)

            if equity_curve:
                peak = equity_curve[0]
                max_dd_val = 0.0
                for v in equity_curve:
                    if v > peak:
                        peak = v
                    if peak > 0:
                        dd = 1.0 - v / peak
                        if dd > max_dd_val:
                            max_dd_val = dd
                max_dd = max_dd_val

            if max_dd and max_dd > 0 and not math.isnan(annual_twr):
                calmar = annual_twr / max_dd

        if cashflows:
            irr = compute_irr(cashflows)

        first_date = rows[0]["date"] if rows else ""
        last_date = rows[-1]["date"] if rows else ""

        def fmt(x: float) -> str:
            if x is None or isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
                return ""
            return f"{x:.10f}"

        summary_rows.append(
            {
                "trader": trader,
                "num_return_days": str(n_returns),
                "start_date": first_date,
                "end_date": last_date,
                "total_twr": fmt(total_twr),
                "annual_twr_252": fmt(annual_twr),
                "daily_mean_return": fmt(mean_ret),
                "daily_vol": fmt(std_ret),
                "annual_vol": fmt(annual_vol),
                "max_drawdown": fmt(max_dd),
                "sharpe": fmt(sharpe),
                "calmar": fmt(calmar),
                "money_weighted_return_irr": fmt(irr),
            }
        )

    with output_daily_returns_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["trader", "date", "daily_return"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in daily_returns_rows:
            writer.writerow(row)

    with output_summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "trader",
            "num_return_days",
            "start_date",
            "end_date",
            "total_twr",
            "annual_twr_252",
            "daily_mean_return",
            "daily_vol",
            "annual_vol",
            "max_drawdown",
            "sharpe",
            "calmar",
            "money_weighted_return_irr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

