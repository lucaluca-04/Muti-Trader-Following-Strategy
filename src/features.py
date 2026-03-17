from __future__ import annotations

"""
Feature engineering utilities shared across scripts.

This centralizes:
- per-trader features from build_trader_features.py
- per-window equity-slice features from profiling_step5_rolling_backtest.py
"""

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def percentile(arr, p: float) -> float:
    if not arr:
        return float("nan")
    arr = sorted(arr)
    n = len(arr)
    k = int(p * (n - 1))
    return arr[k]


def basic_moments(vals: List[float]):
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = sum(vals) / n
    if n == 1:
        return (mean, 0.0, float("nan"), float("nan"))
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    if std == 0 or n < 3:
        return (mean, std, float("nan"), float("nan"))
    m3 = sum((v - mean) ** 3 for v in vals) / (n - 1)
    m4 = sum((v - mean) ** 4 for v in vals) / (n - 1)
    skew = m3 / (std**3)
    kurt_excess = m4 / (std**4) - 3.0
    return (mean, std, skew, kurt_excess)


def corr(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return float("nan")
    if n == 1:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    vx = sum((xi - mx) ** 2 for xi in x) / (n - 1)
    vy = sum((yi - my) ** 2 for yi in y) / (n - 1)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def rolling_min_sum(vals: List[float], window: int) -> float:
    n = len(vals)
    if n < window or window <= 0:
        return float("nan")
    s = sum(vals[:window])
    min_s = s
    for i in range(window, n):
        s += vals[i] - vals[i - window]
        if s < min_s:
            min_s = s
    return min_s


def build_trader_features(
    raw_csv: str | Path,
    daily_ret_csv: str | Path,
    perf_csv: str | Path,
    output_features_csv: str | Path,
) -> None:
    """
    Library-style wrapper for build_trader_features.py:main.
    """
    raw_csv = Path(raw_csv)
    daily_ret_csv = Path(daily_ret_csv)
    perf_csv = Path(perf_csv)
    output_features_csv = Path(output_features_csv)

    num_days: Dict[str, int] = {}
    with perf_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trader = row["trader"]
            try:
                n = int(row["num_return_days"]) if row["num_return_days"] else 0
            except Exception:
                n = 0
            num_days[trader] = n

    equity_list: Dict[str, List[float]] = defaultdict(list)
    max_netdep: Dict[str, float] = defaultdict(float)
    prev_equity_map: Dict[str, Dict[str, float]] = defaultdict(dict)

    by_trader_rows: Dict[str, List[tuple]] = defaultdict(list)
    with raw_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trader = r.get("trader") or r.get("user")
            if not trader:
                continue
            date = r.get("date")
            if not date:
                continue
            eq = safe_float(r.get("equity"))
            nd = safe_float(r.get("net_deposit"))
            by_trader_rows[trader].append((date, eq, nd))

    for trader, rows in by_trader_rows.items():
        rows.sort(key=lambda x: x[0])
        last_eq = None
        last_nd = 0.0
        for i, (date, eq, nd) in enumerate(rows):
            if not math.isnan(eq):
                equity_list[trader].append(eq)
            if not math.isnan(nd) and nd > max_netdep[trader]:
                max_netdep[trader] = nd
            if i > 0 and last_eq is not None:
                prev_equity_map[trader][date] = last_eq
            last_eq = eq

    avg_equity_all: List[float] = []
    for trader, eqs in equity_list.items():
        if eqs:
            avg_equity_all.append(sum(eqs) / len(eqs))
    avg_equity_threshold = percentile(avg_equity_all, 0.05)

    all_days = list(num_days.values())
    all_days_sorted = sorted(all_days)
    days_q25 = percentile(all_days_sorted, 0.25)
    min_days = max(30, int(days_q25))

    returns_by_trader: Dict[str, List[tuple]] = defaultdict(list)
    with daily_ret_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trader = r["trader"]
            date = r["date"]
            dr = r.get("daily_return")
            if not dr:
                continue
            try:
                v = float(dr)
            except Exception:
                continue
            returns_by_trader[trader].append((date, v))

    feature_rows: List[dict] = []
    for trader, ret_list in returns_by_trader.items():
        if trader not in num_days:
            continue

        n_days = num_days.get(trader, 0)
        eqs = equity_list.get(trader, [])
        avg_eq = sum(eqs) / len(eqs) if eqs else float("nan")
        max_nd = max_netdep.get(trader, float("nan"))

        basic_keep = (
            n_days >= min_days
            and not math.isnan(avg_eq)
            and avg_eq >= avg_equity_threshold
        )

        ret_list.sort(key=lambda x: x[0])
        raw_rets = [safe_float(v) for _, v in ret_list]
        rets: List[float] = []
        for r in raw_rets:
            if math.isnan(r) or not math.isfinite(r):
                continue
            if r > 1.0:
                r = 1.0
            elif r < -1.0:
                r = -1.0
            rets.append(r)

        pos = [r for r in rets if r > 0]
        neg = [r for r in rets if r < 0]
        pos_ratio = len(pos) / len(rets) if rets else float("nan")
        avg_gain = sum(pos) / len(pos) if pos else float("nan")
        avg_loss = sum(neg) / len(neg) if neg else float("nan")
        payoff_ratio = (avg_gain / abs(avg_loss)) if pos and neg and avg_loss != 0 else float("nan")

        mean_r, std_r, skew_r, kurt_r = basic_moments(rets)

        has_volatility = not math.isnan(std_r) and std_r > 0.0
        is_kept = basic_keep and has_volatility

        ACTIVE_EPS = 1e-6
        active_days = sum(1 for _, v in ret_list if abs(safe_float(v)) > ACTIVE_EPS)
        n_ret = len(ret_list)
        active_day_ratio = (active_days / n_ret) if n_ret else 0.0
        max_gap_calendar_days = 0
        if n_ret >= 2:
            def _parse_date(s: str) -> datetime:
                s = s.strip()[:10].replace("/", "-")
                return datetime.strptime(s, "%Y-%m-%d")

            dates_sorted = sorted([(r[0], r[1]) for r in ret_list], key=lambda x: x[0])
            for i in range(len(dates_sorted) - 1):
                d0 = _parse_date(dates_sorted[i][0])
                d1 = _parse_date(dates_sorted[i + 1][0])
                gap = (d1 - d0).days
                if gap > max_gap_calendar_days:
                    max_gap_calendar_days = gap

        sorted_rets = sorted(rets)
        p1 = percentile(sorted_rets, 0.01) if sorted_rets else float("nan")
        p5 = percentile(sorted_rets, 0.05) if sorted_rets else float("nan")
        p95 = percentile(sorted_rets, 0.95) if sorted_rets else float("nan")
        p99 = percentile(sorted_rets, 0.99) if sorted_rets else float("nan")

        tail_count = max(1, int(0.05 * len(sorted_rets))) if sorted_rets else 0
        tail_avg = (
            sum(sorted_rets[:tail_count]) / tail_count if tail_count > 0 else float("nan")
        )

        equity_curve: List[float] = []
        if rets:
            eq = 1.0
            peak = eq
            max_dd = 0.0
            eq_series = [eq]
            for r in rets:
                eq *= 1.0 + r
                eq_series.append(eq)
                if eq > peak:
                    peak = eq
                if peak > 0:
                    dd = 1.0 - eq / peak
                    if dd > max_dd:
                        max_dd = dd
            equity_curve = eq_series
        else:
            max_dd = float("nan")

        worst_5d = rolling_min_sum(rets, 5)
        worst_10d = rolling_min_sum(rets, 10)

        if len(rets) >= 2:
            r_t = rets[1:]
            r_tm1 = rets[:-1]
            acf1 = corr(r_t, r_tm1)
            abs_r_t = [abs(x) for x in r_t]
            abs_r_tm1 = [abs(x) for x in r_tm1]
            acf1_abs = corr(abs_r_t, abs_r_tm1)
        else:
            acf1 = float("nan")
            acf1_abs = float("nan")

        prev_eq_for_date = prev_equity_map.get(trader, {})
        pair_rets: List[float] = []
        pair_eqs: List[float] = []
        for date, r in ret_list:
            peq = prev_eq_for_date.get(date)
            if peq is None or math.isnan(peq):
                continue
            pair_rets.append(r)
            pair_eqs.append(peq)

        if len(pair_rets) >= 2:
            ret_eq_corr = corr(pair_rets, pair_eqs)
        elif has_volatility and len(rets) >= 2:
            ret_eq_corr = 0.0
        else:
            ret_eq_corr = float("nan")

        big_ret: List[float] = []
        small_ret: List[float] = []
        if pair_eqs:
            median_eq = percentile(pair_eqs, 0.5)
            for r, e in zip(pair_rets, pair_eqs):
                if e >= median_eq:
                    big_ret.append(r)
                else:
                    small_ret.append(r)
        mean_ret_big = sum(big_ret) / len(big_ret) if big_ret else float("nan")
        mean_ret_small = sum(small_ret) / len(small_ret) if small_ret else float("nan")

        def fmt(x: float) -> str:
            if x is None or isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
                return ""
            return f"{x:.10f}"

        row = {
            "trader": trader,
            "num_return_days": str(n_days),
            "avg_equity": fmt(avg_eq),
            "max_net_deposit": fmt(max_nd),
            "is_kept": "1" if is_kept else "0",
            "pos_day_ratio": fmt(pos_ratio),
            "avg_gain": fmt(avg_gain),
            "avg_loss": fmt(avg_loss),
            "payoff_ratio": fmt(payoff_ratio),
            "mean_return": fmt(mean_r),
            "std_return": fmt(std_r),
            "skew_return": fmt(skew_r),
            "kurtosis_excess": fmt(kurt_r),
            "p1_return": fmt(p1),
            "p5_return": fmt(p5),
            "p95_return": fmt(p95),
            "p99_return": fmt(p99),
            "tail_avg_5pct": fmt(tail_avg),
            "max_drawdown_synth": fmt(max_dd),
            "worst_5d_return": fmt(worst_5d),
            "worst_10d_return": fmt(worst_10d),
            "acf1_return": fmt(acf1),
            "acf1_abs_return": fmt(acf1_abs),
            "ret_prev_equity_corr": fmt(ret_eq_corr),
            "mean_return_big_equity": fmt(mean_ret_big),
            "mean_return_small_equity": fmt(mean_ret_small),
            "num_active_days": str(active_days),
            "active_day_ratio": fmt(active_day_ratio),
            "max_gap_calendar_days": str(max_gap_calendar_days),
        }
        feature_rows.append(row)

    fieldnames = [
        "trader",
        "num_return_days",
        "avg_equity",
        "max_net_deposit",
        "is_kept",
        "pos_day_ratio",
        "avg_gain",
        "avg_loss",
        "payoff_ratio",
        "mean_return",
        "std_return",
        "skew_return",
        "kurtosis_excess",
        "p1_return",
        "p5_return",
        "p95_return",
        "p99_return",
        "tail_avg_5pct",
        "max_drawdown_synth",
        "worst_5d_return",
        "worst_10d_return",
        "acf1_return",
        "acf1_abs_return",
        "ret_prev_equity_corr",
        "mean_return_big_equity",
        "mean_return_small_equity",
        "num_active_days",
        "active_day_ratio",
        "max_gap_calendar_days",
    ]

    with output_features_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in feature_rows:
            writer.writerow(row)


def compute_features_from_equity_slice(equity: pd.Series) -> dict[str, float]:
    """
    Port of compute_features_from_equity_slice from profiling_step5_rolling_backtest.py,
    so that rolling backtest and static profiling use consistent definitions.
    """
    equity = equity.dropna().sort_index()
    if len(equity) < 2:
        return {c: np.nan for c in [
            "mean_return",
            "std_return",
            "max_drawdown_synth",
            "tail_avg_5pct",
            "p5_return",
            "ret_prev_equity_corr",
            "avg_equity",
            "max_net_deposit",
            "mean_return_big_equity",
            "mean_return_small_equity",
            "acf1_return",
            "acf1_abs_return",
        ]} | {"num_active_days": 0}

    ret = equity.pct_change().dropna()
    ret = ret.replace([np.inf, -np.inf], np.nan)
    ret = ret.clip(-1.0, 1.0)
    ret_valid = ret.dropna()
    if len(ret_valid) < 2:
        return {c: np.nan for c in [
            "mean_return",
            "std_return",
            "max_drawdown_synth",
            "tail_avg_5pct",
            "p5_return",
            "ret_prev_equity_corr",
            "avg_equity",
            "max_net_deposit",
            "mean_return_big_equity",
            "mean_return_small_equity",
            "acf1_return",
            "acf1_abs_return",
        ]} | {"num_active_days": 0}

    mean_r = float(ret_valid.mean())
    std_r = float(ret_valid.std())
    if std_r <= 0:
        std_r = np.nan

    cum = (1 + ret).fillna(0).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak.replace(0, np.nan)
    max_dd = float(dd.min()) if dd.notna().any() else np.nan

    sorted_ret = np.sort(ret_valid.values)
    n = len(sorted_ret)
    tail_n = max(1, int(0.05 * n))
    tail_avg = float(np.mean(sorted_ret[:tail_n]))
    p5 = float(np.nanpercentile(sorted_ret, 5.0))

    r_t = ret_valid.values[1:]
    r_tm1 = ret_valid.values[:-1]
    acf1 = corr(list(r_t), list(r_tm1))
    acf1_abs = corr(list(np.abs(r_t)), list(np.abs(r_tm1)))

    prev_eq = equity.shift(1).dropna()
    common = ret_valid.index.intersection(prev_eq.index)
    if len(common) >= 2:
        ret_eq_corr = corr(
            list(ret_valid.loc[common].values),
            list(prev_eq.loc[common].values),
        )
    else:
        ret_eq_corr = 0.0

    avg_equity = float(equity.mean())
    max_net_deposit = np.nan

    med_eq = equity.median()
    if med_eq == med_eq:
        big_ret = ret_valid[equity.loc[ret_valid.index] >= med_eq]
        small_ret = ret_valid[equity.loc[ret_valid.index] < med_eq]
    else:
        big_ret = pd.Series(dtype=float)
        small_ret = pd.Series(dtype=float)
    mean_ret_big = float(big_ret.mean()) if len(big_ret) > 0 else np.nan
    mean_ret_small = float(small_ret.mean()) if len(small_ret) > 0 else np.nan

    ACTIVE_EPS = 1e-6
    num_active_days = int((ret_valid.abs() > ACTIVE_EPS).sum())

    return {
        "num_active_days": num_active_days,
        "mean_return": mean_r,
        "std_return": std_r if std_r == std_r else np.nan,
        "max_drawdown_synth": max_dd,
        "tail_avg_5pct": tail_avg,
        "p5_return": p5,
        "ret_prev_equity_corr": ret_eq_corr,
        "avg_equity": avg_equity,
        "max_net_deposit": max_net_deposit,
        "mean_return_big_equity": mean_ret_big,
        "mean_return_small_equity": mean_ret_small,
        "acf1_return": acf1,
        "acf1_abs_return": acf1_abs,
    }

