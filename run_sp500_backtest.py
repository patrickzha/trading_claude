"""
标普500 + 多进程并行版 5年回测
================================
股票池从原来写死的50只高beta票换成标普500全量成分股（见
build_sp500_universe.py），数据量增大约10倍，配合按周并行
（ProcessPoolExecutor，每个worker里XGBoost单线程，避免和外层进程级
并行抢核心）来把耗时压回可接受范围。
"""
from __future__ import annotations

import os
import time

from backtest import fetch_market_data, fetch_earnings_calendar, run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics
from us_stock_screener import SECTOR_MAP


def main():
    t0 = time.time()
    n_workers = max(1, os.cpu_count() or 1)
    print(f"股票池大小: {len(SECTOR_MAP)}  并行worker数: {n_workers}")

    print("\n" + "#" * 70)
    print("# 第1步：拉取5年历史数据（标普500全量）")
    print("#" * 70)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = fetch_market_data(days=1825)
    common_idx = c_df.index
    print(f"数据拉取耗时: {time.time()-t0:.1f}s")

    print("\n" + "#" * 70)
    print("# 第2步：拉取财报日历（线程池并发）")
    print("#" * 70)
    t1 = time.time()
    earnings_calendar = fetch_earnings_calendar(list(SECTOR_MAP.keys()), n_workers=16)
    print(f"财报日历拉取耗时: {time.time()-t1:.1f}s")

    print("\n" + "#" * 70)
    print("# 第3步：多进程并行回测（默认配置：两条熔断腿+财报避让+账户熔断）")
    print("#" * 70)
    t2 = time.time()
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=True, earnings_calendar=earnings_calendar,
        n_workers=n_workers, xgb_n_jobs=1,
    )
    print(f"回测耗时: {time.time()-t2:.1f}s ({(time.time()-t2)/60:.1f}分钟)")

    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(common_idx):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    run_backtest_with_metrics(results, benchmark_returns=bench_rets)

    total = time.time() - t0
    print(f"\n全流程总耗时: {total:.1f}s ({total/60:.1f} 分钟)")


if __name__ == "__main__":
    main()
