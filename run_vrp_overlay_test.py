"""
Cycle39大修大补②(续)：第208条的VRP信号通过了这次会话证据最强的Granger
可行性检验，这次测真正的策略集成效果——把"VRP为负(隐含波动率<已实现
波动率，通常对应市场应力骤升)"这个状态做成动量腿的降仓叠加规则，方法论
跟第176/183条完全一致，用2022-2026完整236周动量腿回测复测。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import fetch_market_data

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(weekly_rets):
    rets = np.array(weekly_rets)
    nav = np.cumprod(1 + rets)
    n_years = len(rets) / 52
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


def main():
    with open(f"{SCRATCH}/default_full_results_cache.pkl", "rb") as f:
        mom_results = pickle.load(f)

    dates = [pd.Timestamp(r["date"]) for r in mom_results]
    weekly_rets = [r["weekly_return"] for r in mom_results]

    print("拉取VIX+股票池行情数据计算VRP...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()
    market_ret = c_df.mean(axis=1).pct_change().dropna()
    realized_vol = market_ret.rolling(21).std() * np.sqrt(252) * 100
    vrp = (vix - realized_vol).dropna()
    vrp_negative = vrp < 0

    overlay_rets = []
    n_flagged = 0
    for d, r in zip(dates, weekly_rets):
        prior_days = vrp_negative.index[vrp_negative.index < d]
        if len(prior_days) == 0:
            overlay_rets.append(r)
            continue
        last_day = prior_days[-1]
        is_negative = bool(vrp_negative.loc[last_day])
        if is_negative:
            overlay_rets.append(r * 0.5)
            n_flagged += 1
        else:
            overlay_rets.append(r)

    print(f"共{len(dates)}周，其中{n_flagged}周(前一交易日VRP为负)被降仓到50%")

    b_cagr, b_sharpe, b_mdd = metrics(weekly_rets)
    o_cagr, o_sharpe, o_mdd = metrics(overlay_rets)
    print(f"\n基线(现有动量腿，无VRP叠加): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")
    print(f"+VRP为负降仓叠加: CAGR={o_cagr*100:.2f}%, Sharpe={o_sharpe:.3f}, MaxDD={o_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
