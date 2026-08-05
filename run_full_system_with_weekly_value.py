"""
Cycle21大修大补②：第102条验证了value_rebalance_days=5在三元股票组合层面
的改善，但没有测过接到`full_system.py`(股票+动态债券对冲)之后，完整
系统层面的最终提升有多大——这是"如果真的采纳第102条建议"的直接检验，
把Cycle18-20几项独立验证过的改进第一次串起来看整体效果。full_system.py
已经加了value_rebalance_days透传参数，这里直接用。
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

    r_current_best = run_full_system(price_df, membership, spy_close, mom_results)
    r_weekly_value = run_full_system(price_df, membership, spy_close, mom_results, value_rebalance_days=5)

    print(f"\n=== {label} ===")
    print(f"  现有推荐配置(第67条,价值腿月频): CAGR={r_current_best['CAGR']*100:.2f}%, Sharpe={r_current_best['Sharpe']:.3f}, MaxDD={r_current_best['MaxDD']*100:.2f}%")
    print(f"  +采纳第102条(价值腿改周频): CAGR={r_weekly_value['CAGR']*100:.2f}%, Sharpe={r_weekly_value['Sharpe']:.3f}, MaxDD={r_weekly_value['MaxDD']*100:.2f}%")
