"""
Cycle22大修大补②：用第108条PIT修正后的动量数据，重新验证Cycle18-21里
几个最重要的相对比较类发现是否依然成立——value_rebalance_days=5(第102/
103条)、use_vol_targeting(第90/91/97条)、三条腿相关系数矩阵(第106条)。
理论上这些"A配置vs B配置"的相对比较，因为两边共用同一套(带偏差的)动量
基线，方向性结论应该不受影响，这次直接验证这个预期是否成立。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from full_system import run_full_system

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/pit_corrected_2014_2020_results.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/pit_corrected_2009_2014_results.pkl"),
]

for label, price_cache_path, new_mom_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    with open(new_mom_path, "rb") as f:
        mom = pickle.load(f)

    print(f"\n{'='*20} {label}(用修正后动量数据) {'='*20}")

    # 1. value_rebalance_days=5 vs 默认
    r_default = run_full_system(price_df, membership, spy_close, mom)
    r_weekly = run_full_system(price_df, membership, spy_close, mom, value_rebalance_days=5)
    print(f"  [第102/103条复核] 价值腿月频 Sharpe={r_default['Sharpe']:.3f} vs 周频 Sharpe={r_weekly['Sharpe']:.3f} "
          f"-> 周频领先={'是' if r_weekly['Sharpe']>r_default['Sharpe'] else '否'}")

    # 2. use_vol_targeting
    r_vt_off = run_full_system(price_df, membership, spy_close, mom, use_vol_targeting=False)
    r_vt_on = run_full_system(price_df, membership, spy_close, mom, use_vol_targeting=True)
    print(f"  [第90/91/97条复核] vol_targeting关 Sharpe={r_vt_off['Sharpe']:.3f} vs 开 Sharpe={r_vt_on['Sharpe']:.3f} "
          f"-> 开启领先={'是' if r_vt_on['Sharpe']>r_vt_off['Sharpe'] else '否'}, "
          f"MDD: {r_vt_off['MaxDD']*100:.2f}% vs {r_vt_on['MaxDD']*100:.2f}%")

    # 3. 三条腿相关系数矩阵
    from value_investing import backtest_value_strategy
    from low_vol_investing import backtest_low_vol_strategy
    mom_df = pd.DataFrame(mom)[["date", "net_worth"]]
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
    corr = combined.pct_change().dropna().corr()
    print(f"  [第106条复核] 动量-价值相关系数={corr.loc['momentum','value']:.3f}, 动量-低波动={corr.loc['momentum','low_vol']:.3f}, 价值-低波动={corr.loc['value','low_vol']:.3f}")
