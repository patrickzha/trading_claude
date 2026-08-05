"""
Cycle32小修小补(c)：risk_parity权重方案推广到完整full_system层面(股票三腿+
动态债券对冲)的验证——min_variance(第142条)和hrp(第160条)都测过完整系统
层面的效果，risk_parity缺这一步，是一个明确的覆盖空缺，这次补上，让三个
框架在"集成完整度"这个维度上对齐。
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
    r_rp = run_full_system(price_df, membership, spy_close, mom_results, value_weighting="risk_parity")

    print(f"\n=== {label} ===")
    print(f"  基线(等权): CAGR={r_baseline['CAGR']*100:.2f}%, Sharpe={r_baseline['Sharpe']:.3f}, MaxDD={r_baseline['MaxDD']*100:.2f}%")
    print(f"  +risk_parity权重: CAGR={r_rp['CAGR']*100:.2f}%, Sharpe={r_rp['Sharpe']:.3f}, MaxDD={r_rp['MaxDD']*100:.2f}%")
