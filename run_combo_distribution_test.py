"""
Cycle19小修小补：这次会话所有报告都只用Sharpe(基于均值/标准差)描述风险，
从没检查过收益分布本身的形状。Sharpe假设收益近似正态，如果实际分布有
明显负偏(左尾更长，即偶尔出现远超日常波动的大跌)或高峰度(肥尾，极端
事件比正态分布预期的更频繁)，同样的Sharpe数字掩盖了不同的真实风险，
高Sharpe但肥尾/负偏的策略，实盘体验会比数字看起来更吓人。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats

from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020(含2018Q4暴跌+2020新冠暴跌)", f"{SCRATCH}/historical_2014_2020_cache.pkl",
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

    report = run_combo_strategy(price_df, membership, spy_close, mom_results)
    combo_ret = report["combo_nav"].pct_change().dropna()
    spy_ret = spy_close.reindex(report["combo_nav"].index).ffill().pct_change().dropna() if spy_close is not None else None

    skew = stats.skew(combo_ret)
    kurt = stats.kurtosis(combo_ret)  # excess kurtosis, 正态分布=0
    jb_stat, jb_p = stats.jarque_bera(combo_ret)
    var_5pct = np.percentile(combo_ret, 5)
    worst_5 = combo_ret.nsmallest(5)

    print(f"\n=== {label} ===")
    print(f"  组合日收益: 偏度(skew)={skew:.3f}, 超额峰度(excess kurtosis)={kurt:.3f}")
    print(f"  Jarque-Bera正态性检验: 统计量={jb_stat:.1f}, p值={jb_p:.6f} "
          f"({'拒绝正态分布假设' if jb_p < 0.05 else '不能拒绝正态分布假设'})")
    print(f"  5% VaR(单日): {var_5pct*100:.2f}%")
    print(f"  最差5个交易日: {[f'{r*100:.2f}%' for r in worst_5.values]}")
    if spy_ret is not None:
        spy_skew = stats.skew(spy_ret)
        spy_kurt = stats.kurtosis(spy_ret)
        print(f"  (对照SPY同期: 偏度={spy_skew:.3f}, 超额峰度={spy_kurt:.3f})")
