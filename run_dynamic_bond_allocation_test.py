"""
Cycle12大修大补(续)：第65条发现固定60/40股债组合在加息周期(2022-2026)
失效，因为长久期国债对利率变化本身很敏感。这次测动态配置：用10年期
国债收益率(^TNX)的趋势(比如过去60天是不是在上升)判断加息/降息环境，
利率上升时降低/去掉债券敞口(避免久期风险)，利率下降/平稳时维持股债
配置——这是对第65条静态配置的直接改进尝试，不是又加一条新腿，是让
已经证明有效的股债分散化机制在失效的环境下也能避险。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2004-2009(含危机)": ("2004-06-01", "2009-06-02"),
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}


def _extract_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
    return close.squeeze(axis=1)


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    print("拉取 SPY/TLT/^TNX(10年期国债收益率) 2004-2026全区间数据...")
    data = yf.download(["SPY", "TLT", "^TNX"], start="2004-01-01", end="2026-08-01",
                       progress=False, group_by="ticker", auto_adjust=True)
    spy = data["SPY"]["Close"].dropna()
    tlt = data["TLT"]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()

    for label, (start, end) in PERIODS.items():
        spy_sub = spy.loc[start:end]
        tlt_sub = tlt.loc[start:end]
        tnx_sub = tnx.loc[start:end]
        common = spy_sub.index.intersection(tlt_sub.index).intersection(tnx_sub.index)
        if len(common) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        spy_sub, tlt_sub, tnx_sub = spy_sub.loc[common], tlt_sub.loc[common], tnx_sub.loc[common]

        spy_ret = spy_sub.pct_change()
        tlt_ret = tlt_sub.pct_change()

        # 利率趋势判断：60日国债收益率变化，上升(加息/长端利率走高)就降低债券敞口
        tnx_trend = tnx_sub.diff(60)
        bond_weight = pd.Series(0.4, index=common)
        bond_weight[tnx_trend > 0.3] = 0.1  # 60天内10年期收益率上升超过0.3个百分点，大幅降低债券敞口

        dynamic_ret = (1 - bond_weight) * spy_ret + bond_weight * tlt_ret
        static_ret = 0.6 * spy_ret + 0.4 * tlt_ret

        dynamic_nav = (1 + dynamic_ret.dropna()).cumprod()
        static_nav = (1 + static_ret.dropna()).cumprod()
        spy_nav = (1 + spy_ret.dropna()).cumprod()

        print(f"\n=== {label} (n={len(common)}天) ===")
        n_years = len(dynamic_nav) / 252
        for name, nav in [("动态股债配置", dynamic_nav), ("固定60/40", static_nav), ("纯SPY", spy_nav)]:
            cagr = float(nav.iloc[-1] ** (1 / n_years) - 1)
            mdd = true_mdd(nav.values)
            print(f"  {name}: CAGR={cagr*100:.2f}%, MaxDD={mdd*100:.2f}%")


if __name__ == "__main__":
    main()
