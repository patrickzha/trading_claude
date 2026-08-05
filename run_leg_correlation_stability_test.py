"""
Cycle21小修小补：三条腿(动量/价值/低波动)两两相关系数在三段独立历史区间
上是否稳定——这次会话多处引用过具体的相关系数数字(比如低波动跟价值
0.51-0.74，第55条)来论证分散化/非分散化，但没有专门做过一次跨区间的
系统性汇总对比。用已缓存的净值数据算三段区间的完整相关系数矩阵。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

for label, price_cache_path, mom_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    mom_df = pd.DataFrame(mom_results)[["date", "net_worth"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_series = mom_df.set_index("date")["net_worth"]
    mom_daily = mom_series.reindex(price_df.index, method="ffill").dropna()

    _, val_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
    _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)

    legs = {
        "momentum": mom_daily / mom_daily.iloc[0],
        "value": pd.DataFrame(val_nav, columns=["date", "nav"]).set_index("date")["nav"],
        "low_vol": pd.DataFrame(lv_nav, columns=["date", "nav"]).set_index("date")["nav"],
    }
    combined = pd.DataFrame(legs).dropna()
    daily_rets = combined.pct_change().dropna()
    corr = daily_rets.corr()

    print(f"\n=== {label} ===")
    print(corr.round(3).to_string())
