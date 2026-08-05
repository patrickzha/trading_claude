"""
标普500 + 加正则化，看样本外崩盘是不是过拟合造成的、加正则化能不能缓解。
================================================================
上一轮标普500全量回测发现：样本内 CAGR 23.77%/Sharpe 0.78，样本外(后20%，
从没用来调参的48周) CAGR -29.45%/Sharpe -1.21——衰减幅度远超同期SPY自己
的衰减(0.46->-0.17)，是过拟合信号，不只是"那段行情本身难"。

这里用更保守的树结构（更浅、每叶子最少样本数更高、L1/L2都加大）重跑一遍
同样的5年数据，对比样本外的衰减有没有变小。复用上一轮已经缓存好的数据，
不用重新拉。
"""
from __future__ import annotations

import os
import pickle
import time

from backtest import run_walk_forward_backtest, get_weekly_trading_dates, fetch_earnings_calendar
from optimized_metrics_v4 import run_backtest_with_metrics
from us_stock_screener import SECTOR_MAP

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"


def main():
    t0 = time.time()
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    common_idx = c_df.index
    print(f"复用缓存数据，股票数: {len(c_df.columns)}, 交易日: {len(common_idx)}")

    n_workers = max(1, os.cpu_count() or 1)
    print(f"并行worker数: {n_workers}")

    earnings_calendar = fetch_earnings_calendar(list(SECTOR_MAP.keys()), n_workers=16)

    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
        # 比默认更保守的树结构：更浅、叶子样本数门槛更高、L1/L2都加大
        xgb_max_depth=3, xgb_min_child_weight=8,
        xgb_reg_alpha=0.5, xgb_reg_lambda=5.0, xgb_gamma=0.2,
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
    print(f"\n全流程总耗时: {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
