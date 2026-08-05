"""
Cycle20小修小补(续)：验证combo_strategy.py新加的value_rebalance_days参数
(a)默认None时跟改动前行为完全一致，(b)value_rebalance_days=5时对完整
三元组合(不只是价值腿单独)的Sharpe/CAGR/MDD影响。
"""
from __future__ import annotations

import pickle

from combo_strategy import run_combo_strategy
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

    r_default = run_combo_strategy(price_df, membership, spy_close, mom_results)
    r_weekly_val = run_combo_strategy(price_df, membership, spy_close, mom_results, value_rebalance_days=5)

    print(f"\n=== {label} ===")
    print(f"  默认(价值腿月频): CAGR={r_default['CAGR']*100:.2f}%, Sharpe={r_default['Sharpe']:.3f}, MaxDD={r_default['MaxDD']*100:.2f}%")
    print(f"  价值腿改周频: CAGR={r_weekly_val['CAGR']*100:.2f}%, Sharpe={r_weekly_val['Sharpe']:.3f}, MaxDD={r_weekly_val['MaxDD']*100:.2f}%")
