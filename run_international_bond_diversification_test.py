"""
Cycle20小修小补：第64条已经证明国际股票轮动不能提供真正独立于美股的分散化
(全球股市高度联动，尤其危机时刻)，但从没测过国际政府债券——这是一个
性质不同的资产(不仅有当地利率周期的差异，还有汇率敞口，双重跟美国利率
周期解耦的可能性)。用BWX(非美元G7+国际政府债券ETF，不对冲汇率，2007年
上市)对比IEF(美国国债)，看是否提供额外分散化价值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    print("拉取 BWX(国际政府债券)/IEF(美国国债)/SPY 2008-2026数据...")
    data = yf.download(["BWX", "IEF", "SPY"], start="2008-01-01", end="2026-08-01",
                       progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in ["BWX", "IEF", "SPY"]}
    price_df = pd.DataFrame(closes)
    print(f"数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        corr = rets.corr()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  BWX跟SPY相关系数={corr.loc['BWX','SPY']:.4f}   IEF跟SPY相关系数={corr.loc['IEF','SPY']:.4f}")
        print(f"  BWX跟IEF相关系数={corr.loc['BWX','IEF']:.4f}(两种债券本身是否也高度联动)")

        for w in [0.2, 0.3]:
            blend_bwx = (1 - w) * rets["SPY"] + w * rets["BWX"]
            blend_ief = (1 - w) * rets["SPY"] + w * rets["IEF"]
            nav_bwx = (1 + blend_bwx).cumprod()
            nav_ief = (1 + blend_ief).cumprod()
            n_years = len(nav_bwx) / 252
            cagr_bwx = float(nav_bwx.iloc[-1] ** (1 / n_years) - 1)
            cagr_ief = float(nav_ief.iloc[-1] ** (1 / n_years) - 1)
            print(f"  权重{w:.0%}: SPY+BWX CAGR={cagr_bwx*100:.2f}%,MDD={true_mdd(nav_bwx.values)*100:.2f}%  |  "
                  f"SPY+IEF(对照) CAGR={cagr_ief*100:.2f}%,MDD={true_mdd(nav_ief.values)*100:.2f}%")


if __name__ == "__main__":
    main()
