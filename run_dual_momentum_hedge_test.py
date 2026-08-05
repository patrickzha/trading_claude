"""
Cycle20大修大补①：非股票敞口(现在只能选纯债券、或固定比例债券+黄金)引入
dual momentum动态轮动——每月月初，看IEF/GLD/现金(0%收益代理)过去12个月
减去最近1个月(标准的12-1月动量构造，剔除短期反转)的总回报，选表现最好
的那个作为当月的非股票敞口配置(可能是纯债券、纯黄金、或纯现金)，代替
现在"固定纯债券"或"固定比例债券+黄金"这种静态配置。

保留现有的"利率趋势阈值决定股票/非股票敞口比例"这条已验证机制(第79/86条)
不变，dual momentum只决定"非股票敞口那部分具体买什么"，是一个独立的、
可以叠加测试的新维度，不是推倒重来。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from combo_strategy import run_combo_strategy, true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
BOND_WEIGHT = 0.4
BOND_WEIGHT_HIKE = 0.1
RATE_TREND_THRESHOLD = 0.3
TREND_WINDOW = 60
MOM_LOOKBACK = 252  # 12个月(交易日)
MOM_SKIP = 21  # 剔除最近1个月


def sharpe_of(rets):
    if len(rets) < 5:
        return float("nan")
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))


periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
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

    equity_report = run_combo_strategy(price_df, membership, spy_close, mom_results)
    equity_nav = equity_report["combo_nav"]
    equity_ret = equity_nav.pct_change()

    start = equity_nav.index[0].strftime("%Y-%m-%d")
    end = equity_nav.index[-1].strftime("%Y-%m-%d")
    # 多拉一年历史给12-1月动量算回看
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    data = yf.download(["IEF", "GLD", "^TNX"], start=fetch_start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)
    bond_close = data["IEF"]["Close"].dropna()
    gold_close = data["GLD"]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()

    tnx_trend = tnx.reindex(equity_nav.index).ffill().diff(TREND_WINDOW)
    non_equity_w = pd.Series(BOND_WEIGHT, index=equity_nav.index)
    non_equity_w[tnx_trend > RATE_TREND_THRESHOLD] = BOND_WEIGHT_HIKE

    bond_ret_series = bond_close.pct_change()
    gold_ret_series = gold_close.pct_change()
    bond_mom = bond_close.pct_change(MOM_LOOKBACK - MOM_SKIP).shift(MOM_SKIP)
    gold_mom = gold_close.pct_change(MOM_LOOKBACK - MOM_SKIP).shift(MOM_SKIP)

    # 月度决定非股票敞口买什么(用当月第一个交易日的动量信号，整月不变)
    hedge_choice = pd.Series(index=equity_nav.index, dtype=object)
    month_starts = equity_nav.index.to_period("M").drop_duplicates()
    for period in month_starts:
        month_mask = equity_nav.index.to_period("M") == period
        first_day = equity_nav.index[month_mask][0]
        bm = bond_mom.asof(first_day)
        gm = gold_mom.asof(first_day)
        scores = {"bond": bm if pd.notna(bm) else -np.inf,
                  "gold": gm if pd.notna(gm) else -np.inf,
                  "cash": 0.0}
        choice = max(scores, key=scores.get)
        hedge_choice.loc[month_mask] = choice

    hedge_ret = pd.Series(0.0, index=equity_nav.index)
    hedge_ret[hedge_choice == "bond"] = bond_ret_series.reindex(equity_nav.index).ffill()[hedge_choice == "bond"]
    hedge_ret[hedge_choice == "gold"] = gold_ret_series.reindex(equity_nav.index).ffill()[hedge_choice == "gold"]
    # cash: 0%收益，已经是默认初始化值

    dm_full_ret = ((1 - non_equity_w) * equity_ret + non_equity_w * hedge_ret).dropna()
    dm_full_nav = (1 + dm_full_ret).cumprod()

    fixed_bond_ret = ((1 - non_equity_w) * equity_ret + non_equity_w * bond_ret_series.reindex(equity_nav.index).ffill()).dropna()
    fixed_bond_nav = (1 + fixed_bond_ret).cumprod()

    n_years = len(dm_full_nav) / 252
    dm_cagr = float(dm_full_nav.iloc[-1] ** (1 / n_years) - 1)
    dm_mdd = true_max_drawdown(dm_full_nav.values)
    fb_cagr = float(fixed_bond_nav.iloc[-1] ** (1 / n_years) - 1)
    fb_mdd = true_max_drawdown(fixed_bond_nav.values)

    choice_counts = hedge_choice.value_counts(normalize=True)
    print(f"\n=== {label} ===")
    print(f"  固定纯债券(现有默认): CAGR={fb_cagr*100:.2f}%, Sharpe={sharpe_of(fixed_bond_ret):.3f}, MaxDD={fb_mdd*100:.2f}%")
    print(f"  dual momentum动态轮动: CAGR={dm_cagr*100:.2f}%, Sharpe={sharpe_of(dm_full_ret):.3f}, MaxDD={dm_mdd*100:.2f}%")
    print(f"  月度选择分布: {dict(choice_counts.round(3))}")
