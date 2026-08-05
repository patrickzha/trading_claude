"""
标普400中盘股票池验证——标普500是全球分析师覆盖最密集、定价最有效的股票池，
技术面动量信号在这里天然难找edge；中盘股覆盖度更低，历史上动量因子在中小盘
更有效，这是判断"换股票池"是不是比继续在标普500里抠模型细节更有希望的测试。

用当前已验证的默认配置（止损2%、RSI阈值100、新近度加权26周，全部是模块
默认值，不用显式传）在标普400数据上跑一遍完整统计验证，跟标普500的对应
结果直接比较。sector_map 显式传 sp400 的映射（不能用默认的 SECTOR_MAP，
那是标普500的，中盘票查不到会全部退化成"TECH"，行业分散度检查失去意义）。

不做财报避让（没有单独拉这399只票的财报日历，简化处理，跟2014-2020独立期
验证用同样的简化）。
"""
from __future__ import annotations

import json
import pickle
import os

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp400_data_cache.pkl"


def main():
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    print(f"标普400中盘区间: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df.columns)}只票")

    with open("sp400_universe.json") as f:
        sp400 = json.load(f)
    sector_map = {t: info["sector"] for t, info in sp400["tickers"].items()}

    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,
        sector_map=sector_map,
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

    print("\n[标普400中盘股票池验证报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
