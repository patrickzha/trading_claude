"""
Cycle20小修小补：价值/低波动两条腿现在固定21天(约1个月)调仓，从没测过更
频繁的调仓(比如5天/周频，跟动量腿同频)会不会更好——更频繁调仓理论上能
更及时捕捉候选池变化，但换手成本(cost_bi=0.3%)也会同步增加，两者方向
相反，净效果不确定，值得实测。
"""
from __future__ import annotations

import pickle

import numpy as np

from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

REBALANCE_DAYS = {"周频(5天)": 5, "月频(21天,现有默认)": 21, "季频(63天)": 63}


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
    for rname, rdays in REBALANCE_DAYS.items():
        _, val_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=rdays, top_n=30)
        _, lv_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=rdays, top_n=30)
        val_cagr, val_sharpe, val_mdd = metrics(val_nav)
        lv_cagr, lv_sharpe, lv_mdd = metrics(lv_nav)
        print(f"  {rname}: 价值腿 Sharpe={val_sharpe:.3f},CAGR={val_cagr*100:.2f}%,MDD={val_mdd*100:.2f}%  |  "
              f"低波动腿 Sharpe={lv_sharpe:.3f},CAGR={lv_cagr*100:.2f}%,MDD={lv_mdd*100:.2f}%")
