"""
Cycle67大修大补①：完成第26条留下的悬念——分批建仓(周一/周二/周三跨天各买
1/3，代替周一开盘一次性买满)三窗口完整验证。第26条只在一个20周的诊断
窗口测过，方向出乎意料地正面(分批买入更好)，但明确标注"不满足三窗口
验证标准，不算数"。

这次复用三段区间已缓存的动量腿walk-forward结果(`picks`字段，逐周候选
ticker列表)，避免重新训练模型(每周retrain一次很慢)，用等权假设重新结算
"周一一次性买入" vs "周一二三分批买入"两个版本——如实标注这个简化：原始
`weekly_return`用的是置信度加权仓位，这次为了公平对比只能假设等权(因为
缓存没保留每周的具体权重)，两个版本(一次性/分批)都用同样的等权假设，
比较本身仍然公平，只是绝对收益率数字跟原始加权版本不完全一致。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
STOP_LOSS_PCT = 0.02
COST_BI = 0.003
N_TRANCHES = 3

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", f"{SCRATCH}/default_full_results_cache.pkl"),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]


def onetime_ret(ticker, monday, friday, c_df, o_df, l_df):
    try:
        buy_price = o_df.loc[monday, ticker]
        if pd.isna(buy_price) or buy_price <= 0:
            return 0.0
        stop_price = buy_price * (1 - STOP_LOSS_PCT)
        week_days = c_df.loc[monday:friday, ticker].index
        for day in week_days[1:]:
            day_low = l_df.loc[day, ticker] if day in l_df.index else np.nan
            if pd.notna(day_low) and day_low <= stop_price:
                return (stop_price - buy_price) / buy_price - COST_BI * 0.5
        sell_price = c_df.loc[friday, ticker]
        if pd.isna(sell_price) or sell_price <= 0:
            return 0.0
        return (sell_price - buy_price) / buy_price - COST_BI
    except Exception:
        return 0.0


def staggered_ret(ticker, monday, friday, c_df, o_df, l_df):
    try:
        week_days = c_df.loc[monday:friday, ticker].index
        if len(week_days) < N_TRANCHES + 1:
            return 0.0
        tranche_days = week_days[:N_TRANCHES]
        tranche_prices = o_df.loc[tranche_days, ticker]
        if tranche_prices.isna().any() or (tranche_prices <= 0).any():
            return 0.0
        buy_price = float(tranche_prices.mean())
        stop_price = buy_price * (1 - STOP_LOSS_PCT)
        remaining_days = week_days[N_TRANCHES:]
        for day in remaining_days:
            day_low = l_df.loc[day, ticker] if day in l_df.index else np.nan
            if pd.notna(day_low) and day_low <= stop_price:
                return (stop_price - buy_price) / buy_price - COST_BI * 0.5
        sell_price = c_df.loc[friday, ticker]
        if pd.isna(sell_price) or sell_price <= 0:
            return 0.0
        return (sell_price - buy_price) / buy_price - COST_BI
    except Exception:
        return 0.0


def metrics(rets):
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak)) if len(nav) else 0.0
    return sharpe, mdd, nav[-1] - 1 if len(nav) else 0.0


for label, price_cache_path, mom_cache_path in PERIODS:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, l_df = cache["c"], cache["o"], cache["l"]
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)

    # r["date"]本身就是周一(见backtest.py第610行"date": monday...)，不是
    # 决策日——第一版脚本错误地把它当成决策日再+1找周一，同时用固定5个
    # 交易日找周五，两处都会在遇到假期缩短的周(感恩节/圣诞节/劳动节等)
    # 时错位，用get_weekly_trading_dates()正确处理这些情况。
    week_pairs = {monday.strftime("%Y-%m-%d"): friday for monday, friday in get_weekly_trading_dates(c_df.index)}

    onetime_weekly, staggered_weekly = [], []
    for r in mom_results:
        picks = r.get("picks", [])
        if not picks:
            onetime_weekly.append(0.0)
            staggered_weekly.append(0.0)
            continue
        monday = pd.Timestamp(r["date"])
        friday = week_pairs.get(r["date"])
        if friday is None:
            continue

        week_onetime = [onetime_ret(t, monday, friday, c_df, o_df, l_df) for t in picks if t in c_df.columns]
        week_staggered = [staggered_ret(t, monday, friday, c_df, o_df, l_df) for t in picks if t in c_df.columns]
        onetime_weekly.append(np.mean(week_onetime) if week_onetime else 0.0)
        staggered_weekly.append(np.mean(week_staggered) if week_staggered else 0.0)

    s_one, m_one, c_one = metrics(onetime_weekly)
    s_stag, m_stag, c_stag = metrics(staggered_weekly)
    print(f"\n=== {label} (n={len(onetime_weekly)}周) ===")
    print(f"  一次性买入: Sharpe={s_one:.3f}, MaxDD={m_one*100:.2f}%, 累计收益={c_one*100:.2f}%")
    print(f"  分批买入(周一二三): Sharpe={s_stag:.3f}, MaxDD={m_stag*100:.2f}%, 累计收益={c_stag*100:.2f}%")

print("\n全部完成。")
