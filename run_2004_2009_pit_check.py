"""
Cycle22小修小补：第62条(2004-2009真正危机区间，经典动量+低波动二元组合)
用的也是"当前股票池"拉的历史数据，同样从没做过纳入日期过滤——经典动量
(12-1月价格动量)理论上跟ML动量腿一样，容易受益于"即将被纳入指数的强势
新贵股"这个前视偏差(第109条已经揭示的机制)，这次快速验证一下。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from classic_momentum import backtest_classic_momentum
from low_vol_investing import backtest_low_vol_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"

with open(f"{SCRATCH}/historical_2004_2009_cache.pkl", "rb") as f:
    cache = pickle.load(f)
price_df, spy_close = cache["c"], cache.get("spy")

with open(ADDED_DATES_PATH) as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

flat_membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
pit_membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


_, cm_nav_flat = backtest_classic_momentum(price_df, flat_membership, spy_close, rebalance_days=21, top_n=30)
_, cm_nav_pit = backtest_classic_momentum(price_df, pit_membership, spy_close, rebalance_days=21, top_n=30)
_, lv_nav_flat = backtest_low_vol_strategy(price_df, flat_membership, spy_close, rebalance_days=21, top_n=30)
_, lv_nav_pit = backtest_low_vol_strategy(price_df, pit_membership, spy_close, rebalance_days=21, top_n=30)

cm_f = metrics(cm_nav_flat)
cm_p = metrics(cm_nav_pit)
lv_f = metrics(lv_nav_flat)
lv_p = metrics(lv_nav_pit)

print(f"经典动量  未修正: Sharpe={cm_f[1]:.3f},CAGR={cm_f[0]*100:.2f}%,MDD={cm_f[2]*100:.2f}%   修正后: Sharpe={cm_p[1]:.3f},CAGR={cm_p[0]*100:.2f}%,MDD={cm_p[2]*100:.2f}%")
print(f"低波动    未修正: Sharpe={lv_f[1]:.3f},CAGR={lv_f[0]*100:.2f}%,MDD={lv_f[2]*100:.2f}%   修正后: Sharpe={lv_p[1]:.3f},CAGR={lv_p[0]*100:.2f}%,MDD={lv_p[2]*100:.2f}%")
