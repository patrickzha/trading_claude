"""
Cycle18小修小补：full_system.py的bond_weight(默认40%)/bond_weight_hike
(默认10%)是权重幅度，之前只测过触发阈值(第69/79条)，从没测过这两个幅度
本身对不对。这次在三段有完整组合数据的区间测几组备选幅度。
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

CONFIGS = {
    "默认(40%/10%)": dict(bond_weight=0.4, bond_weight_hike=0.1),
    "更保守(60%/20%)": dict(bond_weight=0.6, bond_weight_hike=0.2),
    "更激进(20%/5%)": dict(bond_weight=0.2, bond_weight_hike=0.05),
    "无降仓区别(40%/40%,即取消动态开关只留固定40%债券)": dict(bond_weight=0.4, bond_weight_hike=0.4),
}

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

    print(f"\n=== {label} ===")
    for cname, kwargs in CONFIGS.items():
        r = run_full_system(price_df, membership, spy_close, mom_results, **kwargs)
        print(f"  {cname}: CAGR={r['CAGR']*100:.2f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")
