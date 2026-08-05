"""
Cycle25大修大补②：第117/125/126条揭示的"phantom候选"(CVNA/TSLA/CRWD这类
未纳入指数就被误选的票)有一个共同特点——被选中时往往还是相对年轻、市值
和流动性不如成熟蓝筹股的成长股。这次测一个不同的过滤维度：不管有没有
被纳入标普500，直接按"平均日成交额"(价格×成交量，衡量流动性/间接反映
规模)过滤候选池，只保留流动性最高的300只，看结果会不会变化——如果
"追涨小盘高波动股"本身就是这套动量模型收益的重要来源(不只是PIT那部分
偏差)，限制到流动性更高、更成熟的大盘股应该会让收益进一步下降；如果
不是，结果应该差不多。

用PIT修正后的候选池(已经排除了未纳入指数的票)做基础，这次测的是一个
独立的、新的维度，不是重复第108-127条的调查。
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")
ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"
N_WORKERS = max(1, os.cpu_count() or 1)
TOP_N_LIQUID = 300


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    t0 = time.time()
    with open(f"{SCRATCH}/historical_2014_2020_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]

    dollar_vol = (c_df * v_df).mean()
    liquid_tickers = dollar_vol.sort_values(ascending=False).head(TOP_N_LIQUID).index.tolist()
    print(f"按平均日成交额排名，取流动性最高的{TOP_N_LIQUID}/{len(c_df.columns)}只票")
    print(f"被剔除的低流动性票样例: {[t for t in c_df.columns if t not in liquid_tickers][:15]}")

    c_liq, o_liq, h_liq, l_liq, v_liq = c_df[liquid_tickers], o_df[liquid_tickers], h_df[liquid_tickers], l_df[liquid_tickers], v_df[liquid_tickers]

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    with open(ADDED_DATES_PATH) as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}
    membership = compute_point_in_time_membership(c_liq.index, list(c_liq.columns), added_dates, {})

    print("\n跑高流动性子集(300只，同时应用PIT纳入日期过滤，保持两个维度可比)的完整回测...")
    results_liquid = run_walk_forward_backtest(
        c_liq, o_liq, h_liq, l_liq, v_liq, vix,
        enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1, earnings_calendar=earnings_calendar,
        point_in_time_membership=membership,
    )
    m_liquid = rough_metrics(results_liquid)

    with open(f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", "rb") as f:
        results_full = pickle.load(f)
    m_full = rough_metrics(results_full)

    print(f"\n全量PIT修正后候选池(484只): {m_full}")
    print(f"流动性Top{TOP_N_LIQUID}子集: {m_liquid}")
    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
