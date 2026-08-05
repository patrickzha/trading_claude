"""
Cycle21大修大补①的紧急延续：第108条发现动量腿在2014-2020/2009-2014两段
的表现被"未做纳入日期过滤"严重高估，这次立刻检查价值/低波动两条腿是否
有同样的问题——它们从会话早期开始，在这两段区间上一直用的是
`{date: 全部列}`这种没有点位时间过滤的假membership，理论上存在同样方向
的偏差。这次用跟第108条同样的`sp500_membership_added_dates.json`重新
生成正确的membership，重跑价值/低波动两条腿，量化影响。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl"),
]


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


with open(ADDED_DATES_PATH) as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

for label, price_cache_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")

    old_membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    new_membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

    print(f"\n=== {label} ===")
    _, val_nav_old = backtest_value_strategy(price_df, old_membership, spy_close, rebalance_days=21, top_n=30)
    _, val_nav_new = backtest_value_strategy(price_df, new_membership, spy_close, rebalance_days=21, top_n=30)
    _, lv_nav_old = backtest_low_vol_strategy(price_df, old_membership, spy_close, rebalance_days=21, top_n=30)
    _, lv_nav_new = backtest_low_vol_strategy(price_df, new_membership, spy_close, rebalance_days=21, top_n=30)

    vo_cagr, vo_sharpe, vo_mdd = metrics(val_nav_old)
    vn_cagr, vn_sharpe, vn_mdd = metrics(val_nav_new)
    lo_cagr, lo_sharpe, lo_mdd = metrics(lv_nav_old)
    ln_cagr, ln_sharpe, ln_mdd = metrics(lv_nav_new)

    print(f"  价值腿  修正前: Sharpe={vo_sharpe:.3f},CAGR={vo_cagr*100:.2f}%,MDD={vo_mdd*100:.2f}%   修正后: Sharpe={vn_sharpe:.3f},CAGR={vn_cagr*100:.2f}%,MDD={vn_mdd*100:.2f}%")
    print(f"  低波动腿修正前: Sharpe={lo_sharpe:.3f},CAGR={lo_cagr*100:.2f}%,MDD={lo_mdd*100:.2f}%   修正后: Sharpe={ln_sharpe:.3f},CAGR={ln_cagr*100:.2f}%,MDD={ln_mdd*100:.2f}%")
