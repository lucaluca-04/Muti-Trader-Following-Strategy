from __future__ import annotations

"""
Data I/O helpers shared across all scripts.

Centralises:
- safe_filename for trader addresses
- load_equity_curve / load_all_curves used by profiling step3/4 and rolling backtest
"""

import csv
import os
from pathlib import Path

import pandas as pd


def safe_filename(trader: str) -> str:
    """Use trader address as filename; replace filesystem-unsafe characters."""
    return trader.replace("/", "_").replace("\\", "_") + ".csv"


def load_equity_curve(curves_dir: Path, trader: str) -> pd.DataFrame | None:
    """
    Load date, equity_twr from <curves_dir>/<trader>.csv.
    Returns a DataFrame sorted by date, or None if file is missing / malformed.
    """
    path = curves_dir / safe_filename(trader)
    if not path.is_file():
        # try bare name (legacy)
        path = curves_dir / f"{trader}.csv"
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        if "date" not in df.columns or "equity_twr" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "equity_twr"])
        df["equity_twr"] = pd.to_numeric(df["equity_twr"], errors="coerce")
        df = df.dropna(subset=["equity_twr"])
        return df[["date", "equity_twr"]].sort_values("date").drop_duplicates(subset=["date"])
    except Exception:
        return None


def load_all_curves(curves_dir: Path) -> dict[str, pd.Series]:
    """
    Load all trader equity curves from a directory.
    Returns dict: trader_id -> pd.Series(equity_twr) indexed by date.
    """
    out: dict[str, pd.Series] = {}
    for path in sorted(curves_dir.glob("*.csv")):
        trader = path.stem
        try:
            df = pd.read_csv(path)
            if "date" not in df.columns or "equity_twr" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "equity_twr"])
            df["equity_twr"] = pd.to_numeric(df["equity_twr"], errors="coerce")
            df = (
                df.dropna(subset=["equity_twr"])
                .drop_duplicates(subset=["date"])
                .set_index("date")
                .sort_index()
            )
            if len(df) >= 2:
                out[trader] = df["equity_twr"]
        except Exception:
            continue
    return out


def export_equity_curves(
    daily_ret_csv: str | Path,
    output_dir: str | Path,
) -> int:
    """
    Build per-trader TWR equity curves from daily_returns CSV.
    Writes one CSV per trader to output_dir.
    Returns number of files written.
    """
    daily_ret_csv = Path(daily_ret_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict

    by_trader: dict[str, list[tuple]] = defaultdict(list)
    with daily_ret_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trader = row["trader"]
            date = row["date"]
            dr = row.get("daily_return", "").strip()
            if not dr:
                continue
            try:
                r = float(dr)
            except ValueError:
                continue
            by_trader[trader].append((date, r))

    count = 0
    for trader, pairs in by_trader.items():
        pairs.sort(key=lambda x: x[0])
        if not pairs:
            continue
        eq = 1.0
        out_rows = []
        for date, r in pairs:
            eq *= 1.0 + r
            out_rows.append({"date": date, "daily_return": r, "equity_twr": round(eq, 10)})
        path = output_dir / safe_filename(trader)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["date", "daily_return", "equity_twr"])
            w.writeheader()
            w.writerows(out_rows)
        count += 1

    return count
