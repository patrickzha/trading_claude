"""
标普500 + 集成(ensemble) —— 用降方差的方式而不是加偏的方式应对过拟合。
================================================================
默认单模型版：样本内 Sharpe 0.78，样本外 Sharpe -1.21（过拟合信号）。
加正则化版：样本内 0.07，样本外 1.04（衰减方向整个翻过来，但全样本
Sharpe置信区间依然跨0，且最大回撤/超额收益等汇总指标全面变差——
不是真的解决了问题，更像是在小样本上换了一种噪声分布）。

这里换一个不同性质的手段：每周训练5个不同随机种子的模型取平均，用集成
的方式压低单模型自身的方差，而不是用正则化去改变模型的偏——如果样本内
依然能保持不错的Sharpe、同时样本外不再是断崖式下跌，才说明这是真的在
提升稳健性，而不是又一次的参数巧合。
"""
from __future__ import annotations

import os
import json
import pickle
import time

from backtest import run_walk_forward_backtest, get_weekly_trading_dates, fetch_earnings_calendar
from optimized_metrics_v4 import run_backtest_with_metrics
from us_stock_screener import SECTOR_MAP

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"


def main():
    t0 = time.time()
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    common_idx = c_df.index
    print(f"复用缓存数据，股票数: {len(c_df.columns)}, 交易日: {len(common_idx)}")

    n_workers = max(1, os.cpu_count() or 1)
    print(f"并行worker数: {n_workers}")

    if os.path.exists(EARNINGS_CACHE):
        with open(EARNINGS_CACHE, "rb") as f:
            earnings_calendar = pickle.load(f)
        print(f"复用缓存的财报日历: {len(earnings_calendar)}只票")
    else:
        earnings_calendar = fetch_earnings_calendar(list(SECTOR_MAP.keys()), n_workers=16)
        with open(EARNINGS_CACHE, "wb") as f:
            pickle.dump(earnings_calendar, f)

    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        xgb_ensemble_size=5,
    )
    print(f"回测耗时: {(time.time()-t0)/60:.1f}分钟")

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

    with open("/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n全流程总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
