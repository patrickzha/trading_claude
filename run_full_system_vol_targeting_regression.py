"""
Cycle20小修小补：验证full_system.py新加的use_vol_targeting参数(a)默认False
时跟改动前行为完全一致(回归测试)，(b)True时的三段区间结果，跟独立脚本
run_vol_targeting_test.py(第91条，敞口上限已改回100%)的数字对得上。
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

    r_off = run_full_system(price_df, membership, spy_close, mom_results, use_vol_targeting=False)
    r_on = run_full_system(price_df, membership, spy_close, mom_results, use_vol_targeting=True)

    print(f"\n=== {label} ===")
    print(f"  use_vol_targeting=False(默认): CAGR={r_off['CAGR']*100:.2f}%, Sharpe={r_off['Sharpe']:.3f}, MaxDD={r_off['MaxDD']*100:.2f}%")
    print(f"  use_vol_targeting=True: CAGR={r_on['CAGR']*100:.2f}%, Sharpe={r_on['Sharpe']:.3f}, MaxDD={r_on['MaxDD']*100:.2f}%")
