"""
Cycle35大修大补①(续)：第182条通过了feasibility gate，这次测真正的策略
集成效果——把收益率曲线倒挂信号做成动量腿的降仓叠加规则，用2022-2026
完整236周动量腿回测复测，方法论跟第176条(VIX期限结构backwardation)
完全一致，方便直接对比。
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

    print("拉取^TNX/^IRX数据用于倒挂标记...")
    rate_data = yf.download(["^TNX", "^IRX"], start=dates[0] - pd.Timedelta(days=10),
                             end=dates[-1] + pd.Timedelta(days=10),
                             progress=False, group_by="ticker", auto_adjust=True)
    tnx = rate_data["^TNX"]["Close"].dropna()
    irx = rate_data["^IRX"]["Close"].dropna()
    common = tnx.index.intersection(irx.index)
    inverted = (tnx.loc[common] - irx.loc[common]) < 0

    overlay_rets = []
    n_flagged = 0
    for d, r in zip(dates, weekly_rets):
        prior_days = inverted.index[inverted.index < d]
        if len(prior_days) == 0:
            overlay_rets.append(r)
            continue
        last_day = prior_days[-1]
        is_inverted = bool(inverted.loc[last_day])
        if is_inverted:
            overlay_rets.append(r * 0.5)
            n_flagged += 1
        else:
            overlay_rets.append(r)

    print(f"共{len(dates)}周，其中{n_flagged}周(前一交易日倒挂)被降仓到50%")

    b_cagr, b_sharpe, b_mdd = metrics(weekly_rets)
    o_cagr, o_sharpe, o_mdd = metrics(overlay_rets)
    print(f"\n基线(现有动量腿，无倒挂叠加): CAGR={b_cagr*100:.2f}%, Sharpe={b_sharpe:.3f}, MaxDD={b_mdd*100:.2f}%")
    print(f"+倒挂降仓叠加: CAGR={o_cagr*100:.2f}%, Sharpe={o_sharpe:.3f}, MaxDD={o_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
