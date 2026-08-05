"""
Cycle64大修大补②(续)：DAX40个股动量策略可行性验证。用当前已验证的默认
配置(止损2%、RSI阈值100、新近度加权26周)完整跑一遍，对照基准是DAX指数
本身(不是SPY，德国股票没有理由要跟美股基准比)。

如实标注局限：(1)用当前"现在"的DAX40成分股名单，没有做点位时间修正，
跟第20条标普400测试同等简化程度，存在生存偏差；(2)宏观熔断信号复用
美股VIX(没有现成的德国波动率指数数据源)，理论上不是最合适的信号，但
这次是可行性初筛，不是最终判断，如实标注不修正。
"""
from __future__ import annotations

import json
import os
import pickle

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/dax_data_cache.pkl"


def main():
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, dax_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    print(f"DAX40区间: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df.columns)}只票")

    with open("dax_universe.json") as f:
        dax = json.load(f)
    sector_map = {t: info["sector"] for t, info in dax["tickers"].items()}

    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False,
        sector_map=sector_map,
    )

    bench_rets = None
    if dax_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((dax_close.loc[friday] - dax_close.loc[monday]) / dax_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)

    print("\n[DAX40股票池验证报告，基准=DAX指数本身]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
