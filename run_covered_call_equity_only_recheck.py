"""
Cycle68：第299/301条的备兑看涨期权测试(`run_covered_call_test.py`)把期权
overlay套在`full_nav`(股票+债券60/40混合后的净值)上，这在经济含义上是
错的——备兑看涨期权的前提是"持有正股才能卖出对应的看涨期权"，不能对着
一份"股债混合账户价值"抽象地卖期权，真实世界里没有对应的可执行操作。

把它正式集成进`full_system.py`(`use_covered_call`参数，只作用在股票敞口
`equity_ret`上，债券部分不受影响)之后回归测试时发现，跟原始测试的结果
方向不一致(等权混合版本更差)，这次直接用三段区间对比两种实现方式，
搞清楚原始发现是不是一个方法论错误导致的假象。
"""
from __future__ import annotations

import pickle

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

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

for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    vix = cache.get("vix")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n{'='*70}\n {label}\n{'='*70}")
    r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
    print(f"  无备兑(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%")

    if vix is None:
        print("  [警告] 无VIX数据，跳过")
        continue

    r_call = run_full_system(price_df, membership, spy_close, mom_results,
                              use_covered_call=True, vix_series=vix, **BASE)
    print(f"  备兑看涨(仅作用于股票敞口，正确实现): Sharpe={r_call['Sharpe']:.3f}, MaxDD={r_call['MaxDD']*100:.2f}%")

print("\n全部完成。")
