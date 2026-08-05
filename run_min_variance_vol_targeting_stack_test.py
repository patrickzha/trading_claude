"""
Cycle29大修大补①：第140条(最小方差权重，价值腿个股仓位配置)和第90/91/97条
(波动率目标化，股票腿整体敞口伸缩)是这次会话仅有的两个真正框架级、且
三段区间验证通过的改进——但从来没有测过两者叠加使用的效果。理论上两者
作用在不同层级(一个是"候选池内部怎么分配权重"，一个是"整个股票腿敞口
多大")，应该互补不冲突，这次直接在full_system.py层面测四种组合。
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
    ("①基线(等权+vol_targeting关)", dict(value_weighting="equal", use_vol_targeting=False)),
    ("②只加min_variance", dict(value_weighting="min_variance", use_vol_targeting=False)),
    ("③只加vol_targeting", dict(value_weighting="equal", use_vol_targeting=True)),
    ("④两者叠加", dict(value_weighting="min_variance", use_vol_targeting=True)),
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

    print(f"\n=== {label} ===")
    for cname, kwargs in CONFIGS:
        r = run_full_system(price_df, membership, spy_close, mom_results, **kwargs)
        print(f"  {cname}: CAGR={r['CAGR']*100:.2f}%, Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")
