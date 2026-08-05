"""
Cycle18大修大补①的延续：full_system.py新加了gold_split参数(默认0，向后
兼容)，这里验证(a) gold_split=0跟改动前完全一致(回归测试)，(b)
gold_split=0.5(非股票敞口一半债券一半黄金)在三段有完整组合数据的区间
表现如何，依据是第84条黄金四段历史验证的正面结果。
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

    r0 = run_full_system(price_df, membership, spy_close, mom_results, gold_split=0.0)
    r50 = run_full_system(price_df, membership, spy_close, mom_results, gold_split=0.5)
    r100 = run_full_system(price_df, membership, spy_close, mom_results, gold_split=1.0)

    print(f"\n=== {label} ===")
    print(f"  gold_split=0.0(纯债券,现有默认): CAGR={r0['CAGR']*100:.2f}%, Sharpe={r0['Sharpe']:.3f}, MaxDD={r0['MaxDD']*100:.2f}%")
    print(f"  gold_split=0.5(债券+黄金各半): CAGR={r50['CAGR']*100:.2f}%, Sharpe={r50['Sharpe']:.3f}, MaxDD={r50['MaxDD']*100:.2f}%")
    print(f"  gold_split=1.0(纯黄金): CAGR={r100['CAGR']*100:.2f}%, Sharpe={r100['Sharpe']:.3f}, MaxDD={r100['MaxDD']*100:.2f}%")
