"""
Cycle21小修小补：第102/103条发现价值腿周频调仓全面更好，但周频调仓换手
次数天然是月频的~4倍，现有cost_bi=0.3%假设下已经算过净值(已经扣了成本)，
这次专门测更高的成本假设(0.5%/1.0%)下，周频调仓的优势会不会被更高的
换手成本吃掉——这是对一个"即将被采纳的建议"做压力测试，不是简单假设
默认成本假设一定合理。
"""
from __future__ import annotations

import pickle

from value_investing import backtest_value_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
import numpy as np

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

COST_LEVELS = [0.003, 0.005, 0.01]


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n=== {label} ===")
    for cost in COST_LEVELS:
        _, weekly_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=5, top_n=30, cost_bi=cost)
        _, monthly_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30, cost_bi=cost)
        w_cagr, w_sharpe, w_mdd = metrics(weekly_nav)
        m_cagr, m_sharpe, m_mdd = metrics(monthly_nav)
        print(f"  cost_bi={cost:.1%}: 周频 Sharpe={w_sharpe:.3f},MDD={w_mdd*100:.2f}%  |  "
              f"月频 Sharpe={m_sharpe:.3f},MDD={m_mdd*100:.2f}%  |  周频领先={'是' if w_sharpe>m_sharpe else '否'}")
