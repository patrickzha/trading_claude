"""
Cycle21延续：用第108条修正后的动量结果(pit_corrected_2014_2020/2009_2014_
results.pkl)重跑combo_strategy/full_system，给出"完整PIT修正后"的最终
可信数字——这是第67条"最终推荐配置"在生存偏差修正后的真实样貌，也是
判断Cycle18-21其他发现受多大影响的关键参照。
"""
from __future__ import annotations

import pickle

from combo_strategy import run_combo_strategy
from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl",
     f"{SCRATCH}/pit_corrected_2014_2020_results.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl",
     f"{SCRATCH}/pit_corrected_2009_2014_results.pkl"),
]

for label, price_cache_path, old_mom_path, new_mom_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    with open(old_mom_path, "rb") as f:
        old_mom = pickle.load(f)
    with open(new_mom_path, "rb") as f:
        new_mom = pickle.load(f)

    combo_old = run_combo_strategy(price_df, membership, spy_close, old_mom)
    combo_new = run_combo_strategy(price_df, membership, spy_close, new_mom)

    full_old = run_full_system(price_df, membership, spy_close, old_mom)
    full_new = run_full_system(price_df, membership, spy_close, new_mom)

    print(f"\n=== {label} ===")
    print(f"  三元组合(动量腿修正前): CAGR={combo_old['CAGR']*100:.2f}%, Sharpe={combo_old['Sharpe']:.3f}, MaxDD={combo_old['MaxDD']*100:.2f}%")
    print(f"  三元组合(动量腿修正后): CAGR={combo_new['CAGR']*100:.2f}%, Sharpe={combo_new['Sharpe']:.3f}, MaxDD={combo_new['MaxDD']*100:.2f}%")
    print(f"  完整系统(动量腿修正前): CAGR={full_old['CAGR']*100:.2f}%, Sharpe={full_old['Sharpe']:.3f}, MaxDD={full_old['MaxDD']*100:.2f}%")
    print(f"  完整系统(动量腿修正后): CAGR={full_new['CAGR']*100:.2f}%, Sharpe={full_new['Sharpe']:.3f}, MaxDD={full_new['MaxDD']*100:.2f}%")
