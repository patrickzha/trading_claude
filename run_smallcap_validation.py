"""
Cycle60大修大补②：标普600小盘股票池验证——第20条测过标普400中盘股(结果
标普500全面更好)，这次测真正的小盘股，用当前已验证的动量默认配置(止损2%、
RSI阈值100、新近度加权26周，模块默认值，不显式传)完整跑一遍统计验证。

如实标注局限(跟第20条同样的简化)：sector_map用"当前"标普600成分股名单，
不是点位时间正确的历史成分股(没有单独构建这399→约600只票的纳入/剔除历史)，
存在生存偏差，绝对数字应该打折看待，但作为"这个股票池粒度值不值得投入
更多"的方向性初步判断依然成立——这是第20条已经用过的两步走方法论(先用
简化版初筛，通过了再决定要不要投入构建完整点位时间正确版本)。
"""
from __future__ import annotations

import json
import os
import pickle

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp600_data_cache.pkl"


def main():
    with open(DATA_CACHE, "rb") as f:
        d = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = d["c"], d["o"], d["h"], d["l"], d["v"], d["vix"], d["spy"]
    print(f"标普600小盘区间: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df.columns)}只票")

    with open("sp600_universe.json") as f:
        sp600 = json.load(f)
    sector_map = {t: info["sector"] for t, info in sp600["tickers"].items()}

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

    with open("/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp600_momentum_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("\n[标普600小盘股票池验证报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
