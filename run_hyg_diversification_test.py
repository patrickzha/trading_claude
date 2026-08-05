"""
Cycle63大修大补②(框架改动，新资产类别——信用利差敞口)：高收益公司债(HYG)
作为债券之外新的独立分散化来源，测试逻辑跟gold_split/dollar_split(第84/
206条)完全一样——之前HYG只作为`use_macro_features`里"HYG-LQD信用利差"
这个训练特征的输入(第14条，方向不一致不建议开)，从未作为独立的资产配置
敞口(拿一部分"非股票敞口"直接买HYG，不是用它的利差信息去调整选股模型)
测试过，这是两种完全不同的使用方式：一个是用它的信息含量辅助模型训练，
一个是直接持有它作为收益/风险来源。

机制上的预期：HYG(高收益债)本质上是"债券+信用风险"的混合体，久期比IEF
短、但信用利差在risk-off时会走阔(价格下跌)，理论上跟股票的相关性应该
比IEF更高(信用风险跟股权风险有共同的驱动因素)，分散化效果可能不如纯
利率债，这次直接测试验证这个预期，而不是假设。
"""
from __future__ import annotations

import pickle

import numpy as np

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from stats_utils import to_weekly

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

BASE = dict(value_weighting="hrp", value_rebalance_days=5,
            value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership


print(f"{'='*70}\n HYG(高收益公司债)作为独立分散化敞口，三段区间测试\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for hs in [0.0, 0.2, 0.3, 0.5]:
        r = run_full_system(price_df, membership, spy_close, mom_results, hyg_split=hs, **BASE)
        print(f"  hyg_split={hs}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%, CAGR={r['CAGR']*100:.2f}%")

print("\n全部完成。")
