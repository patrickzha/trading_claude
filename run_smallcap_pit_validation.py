"""
Cycle61大修大补①(续)：用部分PIT修正(纳入日期过滤，见build_sp600_added_dates.py
的局限说明)重跑标普600小盘股票池的动量腿验证，跟第277条的未修正版本对比。
"""
from __future__ import annotations

import json
import os
import pickle

import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
DATA_CACHE = f"{SCRATCH}/sp600_data_cache.pkl"


def main():
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    print(f"标普600小盘区间: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df.columns)}只票")

    with open("sp600_universe.json") as f:
        sp600 = json.load(f)
    sector_map = {t: info["sector"] for t, info in sp600["tickers"].items()}

    with open("sp600_added_dates.json") as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(v) for t, v in added_dates_raw.items()}
    matched = sum(1 for t in c_df.columns if t in added_dates)
    print(f"纳入日期记录覆盖: {matched}/{len(c_df.columns)}只票有记录(没有记录的票视为一直有效，向后兼容)")

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, {})
    avg_pool = sum(len(v) for v in membership.values()) / len(membership)
    print(f"PIT修正后平均每周候选池大小: {avg_pool:.1f}/{len(c_df.columns)}")

    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,
        sector_map=sector_map,
        point_in_time_membership=membership,
    )

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    print("\n[标普600小盘股票池PIT修正后验证报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
