"""
Cycle34大修大补①(续)：第175条通过了feasibility gate(backwardation状态
切换有格兰杰因果支持)，这次测试真正的策略层面问题——把这个信号做成一个
额外的熔断/降仓规则叠加在动量腿上，能不能带来真实的回测改善，而不只是
统计上的领先关系(第175条已经提醒过这两者不是一回事，第1条现货VIX就是
"有一定交互作用但单独看没有因果关系"的先例)。

设计：对于decision_date前一交易日出现backwardation(VIX9D>VIX3M)的那些
周，把当周仓位打对折(不是完全清仓，因为backwardation只在12%的交易日
出现，完全清仓可能错过太多正常收益周)，对比原始净值路径。复用已有的
2022-2026完整系统动量腿回测缓存(`default_full_results_cache.pkl`)，不
重新跑walk-forward训练(节省~30分钟)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

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

    print("拉取VIX9D/VIX3M数据用于backwardation标记...")
    term_data = yf.download(["^VIX9D", "^VIX3M"], start=dates[0] - pd.Timedelta(days=10),
                             end=dates[-1] + pd.Timedelta(days=10),
                             progress=False, group_by="ticker", auto_adjust=True)
    vix9d = term_data["^VIX9D"]["Close"].dropna()
    vix3m = term_data["^VIX3M"]["Close"].dropna()
    backwardation = (vix9d > vix3m)

    overlay_rets = []
    n_flagged = 0
    for d, r in zip(dates, weekly_rets):
        prior_days = backwardation.index[backwardation.index < d]
        if len(prior_days) == 0:
            overlay_rets.append(r)
            continue
        last_day = prior_days[-1]
        is_backwardation = bool(backwardation.loc[last_day])
        if is_backwardation:
            overlay_rets.append(r * 0.5)
            n_flagged += 1
        else:
            overlay_rets.append(r)

    print(f"共{len(dates)}周，其中{n_flagged}周(前一交易日backwardation)被降仓到50%")

    b_cagr, b_sharpe, b_mdd = metrics(weekly_rets)
    o_cagr, o_sharpe, o_mdd = metrics(overlay_rets)
    print(f"\n基线(现有动量腿，无backwardation叠加): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")
    print(f"+backwardation降仓叠加: CAGR={o_cagr*100:.2f}%, Sharpe={o_sharpe:.3f}, MaxDD={o_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
