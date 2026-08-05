"""
Cycle19大修大补②：组合层面波动率目标化(volatility targeting)——一个这次
会话完全没测过的新风控框架。现有的两层风控都是"事后反应"：账户级回撤
熔断(第87/待补)是已经亏了才降仓，动态债券对冲(第65-69/79/86条)是根据
利率趋势换资产。波动率目标化是第三种、机制不同的思路：不管涨跌方向，
只要近期已实现波动率超过目标水平就主动降低敞口(反之提高敞口，有上限)，
本质是控制"波动本身"而不是"已经发生的亏损"或"宏观趋势判断"，是CTA/风险
平价基金常用的经典技术。

方法：对`combo_strategy.py`算出来的三腿等权净值，用trailing 20日已实现
波动率(只用T-1及之前信息，不用当天)算年化波动，目标年化波动15%，敞口=
target_vol/trailing_vol，敞口上限150%(不能加过大杠杆)、下限30%(留基础
敞口，不完全空仓)。超过100%的部分视为用现金杠杆(这里简化为直接线性放大
收益，不单独建模融资成本，如实标注这个简化)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy, true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

TARGET_VOL = 0.15
VOL_WINDOW = 20
EXPOSURE_MIN = 0.3
EXPOSURE_MAX = 1.0

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020(含2018Q4暴跌+2020新冠暴跌)", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]


def metrics(nav):
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    rets = nav.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav.values)
    return cagr, sharpe, mdd


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
    equity_nav = report["combo_nav"]
    equity_ret = equity_nav.pct_change()

    trailing_vol = equity_ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    exposure = (TARGET_VOL / trailing_vol).clip(EXPOSURE_MIN, EXPOSURE_MAX)
    exposure = exposure.shift(1)  # 用T-1为止的波动率决定T日敞口，不用当天信息

    targeted_ret = (exposure * equity_ret).dropna()
    targeted_nav = (1 + targeted_ret).cumprod()

    raw_cagr, raw_sharpe, raw_mdd = metrics(equity_nav.loc[targeted_nav.index[0]:])
    vt_cagr, vt_sharpe, vt_mdd = metrics(targeted_nav)

    print(f"\n=== {label} ===")
    print(f"  原始等权组合: CAGR={raw_cagr*100:.2f}%, Sharpe={raw_sharpe:.3f}, MaxDD={raw_mdd*100:.2f}%")
    print(f"  波动率目标化(目标{TARGET_VOL:.0%}年化): CAGR={vt_cagr*100:.2f}%, Sharpe={vt_sharpe:.3f}, MaxDD={vt_mdd*100:.2f}%")
    print(f"  平均敞口={exposure.mean():.2f}, 敞口标准差={exposure.std():.2f}, "
          f"触及上限{EXPOSURE_MAX:.0%}天数占比={float((exposure>=EXPOSURE_MAX*0.999).mean()):.1%}, "
          f"触及下限{EXPOSURE_MIN:.0%}天数占比={float((exposure<=EXPOSURE_MIN*1.001).mean()):.1%}")
