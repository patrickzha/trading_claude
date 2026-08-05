"""
Cycle27小修小补：验证full_system.py新集成的value_weighting参数(a)默认
"equal"时行为不变，(b)"min_variance"时完整系统层面的效果，(c)跟第102/103
条已验证的value_rebalance_days=5叠加使用时的效果——这是这次会话第一次
把两个都验证过的"价值腿"改进(调仓频率+仓位分配框架)放在一起测。
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

    r_baseline = run_full_system(price_df, membership, spy_close, mom_results)
    r_mv = run_full_system(price_df, membership, spy_close, mom_results, value_weighting="min_variance")
    r_both = run_full_system(price_df, membership, spy_close, mom_results,
                             value_rebalance_days=5, value_weighting="min_variance")

    print(f"\n=== {label} ===")
    print(f"  基线(等权,月频): CAGR={r_baseline['CAGR']*100:.2f}%, Sharpe={r_baseline['Sharpe']:.3f}, MaxDD={r_baseline['MaxDD']*100:.2f}%")
    print(f"  +最小方差权重(月频): CAGR={r_mv['CAGR']*100:.2f}%, Sharpe={r_mv['Sharpe']:.3f}, MaxDD={r_mv['MaxDD']*100:.2f}%")
    print(f"  +最小方差权重+周频调仓(叠加两项改进): CAGR={r_both['CAGR']*100:.2f}%, Sharpe={r_both['Sharpe']:.3f}, MaxDD={r_both['MaxDD']*100:.2f}%")
