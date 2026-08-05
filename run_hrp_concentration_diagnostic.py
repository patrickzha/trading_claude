"""
Cycle64大修大补①(意外发现，从"更新展示脚本"这个日常任务里挖出来的真实
问题)：更新`generate_current_recommendations.py`时发现2026-07-29这个
决策日，价值腿HRP权重给AES一只票分配了83.56%的权重——排查确认不是
脚本bug(AES这60天窗口内已实现波动率std=0.00275，比其余候选低5-10倍，
是真实的价格数据，不是停牌/数据异常)，而是`_hrp_weights()`本身没有任何
单只票权重上限，遇到波动率极端离群的候选就会顶格分配。这个函数从第159条
引入HRP权重以来，被这次会话所有引用"HRP权重"的正面结果(第159/225/255/
276条等)反复使用过，从未检查过这类极端集中的情况在历史回测里出现的
频率、以及是不是拖累了实际报告的Sharpe/MaxDD——这次直接查。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import select_value_portfolio, _hrp_weights
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

REBALANCE_DAYS = 5  # 跟第276条推荐配置(value_rebalance_days=5)一致
COV_WINDOW = 60

for label, cache_path, is_pit in PERIODS:
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    index = price_df.index
    n = len(index)
    rebalance_idx = 252 + COV_WINDOW
    max_weights = []
    n_over_50 = 0
    n_over_30 = 0
    n_total = 0

    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]
        picks = [p for p in select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=30)
                 if p in price_df.columns]
        if len(picks) >= 5:
            hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
            rets_window = hist_window.pct_change().dropna()
            valid_cols = rets_window.columns[rets_window.notna().all()]
            if len(valid_cols) >= 5:
                w = _hrp_weights(rets_window[valid_cols])
                max_w = float(w.max())
                max_weights.append(max_w)
                n_total += 1
                if max_w > 0.5:
                    n_over_50 += 1
                if max_w > 0.3:
                    n_over_30 += 1
        rebalance_idx += REBALANCE_DAYS

    arr = np.array(max_weights)
    print(f"\n=== {label} ===")
    print(f"  总调仓次数={n_total}")
    print(f"  单票最大权重: 均值={arr.mean()*100:.1f}%, 中位数={np.median(arr)*100:.1f}%, "
          f"最大值={arr.max()*100:.1f}%")
    print(f"  单票权重>50%的调仓次数: {n_over_50}/{n_total} ({n_over_50/n_total*100:.1f}%)")
    print(f"  单票权重>30%的调仓次数: {n_over_30}/{n_total} ({n_over_30/n_total*100:.1f}%)")

print("\n全部完成。")
