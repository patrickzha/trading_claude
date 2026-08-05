"""
两阶段架构第一阶段(50日均线粗筛)三窗口初筛通过(见README"验证结论"待写)，
在点位时间正确的554只股票池上跑完整5年验证，对比开启/不开启stage1_filter。
"""
from __future__ import annotations

import pickle
import time

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH, EARNINGS_CACHE


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"], cache["spy"]
    )
    sector_map = cache["sector_map"]
    added_dates = cache["added_dates"]
    removal_dates_used = cache["removal_dates"]

    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, removal_dates_used)

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar_full = pickle.load(f)

    import os
    n_workers = max(1, os.cpu_count() or 1)

    def bench(results):
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
        return bench_rets

    t0 = time.time()
    results_base = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar_full,
        sector_map=sector_map,
        point_in_time_membership=membership,
    )
    print(f"\nbaseline全量耗时: {(time.time()-t0)/60:.1f}分钟")
    print("\n[baseline —— 无stage1粗筛]")
    run_backtest_with_metrics(results_base, benchmark_returns=bench(results_base))

    t1 = time.time()
    results_stage1 = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar_full,
        sector_map=sector_map,
        point_in_time_membership=membership,
        use_stage1_filter=True, stage1_ma_window=50,
    )
    print(f"\nstage1_ma50全量耗时: {(time.time()-t1)/60:.1f}分钟")
    print("\n[stage1_ma50 —— 两阶段架构，50日均线粗筛]")
    run_backtest_with_metrics(results_stage1, benchmark_returns=bench(results_stage1))


if __name__ == "__main__":
    main()
