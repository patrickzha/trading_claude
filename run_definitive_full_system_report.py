"""
Cycle23大修大补②：这次会话Cycle18-22验证过好几项独立的改进(生存偏差修正
[第108-110条,现已是默认缓存]、value_rebalance_days=5[第102/103条]、
use_vol_targeting[第90/91/97条]、gold_split[第84/85条])，但从没有一次
把"如果全部一起用"的最终数字算出来过。这次用已经是默认(PIT修正后)的
动量缓存，跑一个矩阵：{不加任何新参数, 只加value_rebalance_days=5,
再加use_vol_targeting, 再加gold_split=0.5}，四档累加，看最终能做到
什么程度，作为这次会话截至目前唯一的"最终、完整、修正后"系统报告。
"""
from __future__ import annotations

import pickle

from full_system import run_full_system
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

CONFIGS = [
    ("① 基线(PIT修正后动量+默认三元组合+固定债券)", {}),
    ("② +value_rebalance_days=5(第102/103条)", {"value_rebalance_days": 5}),
    ("③ 再+use_vol_targeting(第90/91/97条)", {"value_rebalance_days": 5, "use_vol_targeting": True}),
    ("④ 再+gold_split=0.5(第84/85条)", {"value_rebalance_days": 5, "use_vol_targeting": True, "gold_split": 0.5}),
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

    print(f"\n{'='*20} {label}(动量数据已是PIT修正后版本) {'='*20}")
    for cname, kwargs in CONFIGS:
        r = run_full_system(price_df, membership, spy_close, mom_results, **kwargs)
        print(f"  {cname}: CAGR={r['CAGR']*100:.2f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")
