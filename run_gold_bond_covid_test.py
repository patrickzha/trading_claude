"""
Cycle19小修小补：第84/85条已经用2004-2009(信用危机)和2022-2026(加息周期)
两种宏观环境验证过黄金/债券分散化，唯独没有专门看过2020年3月新冠崩盘这种
第三种类型的冲击——极短、极快、流动性驱动(不是信用收缩、也不是加息)。
这次单独测这段窗口，看黄金/债券在"流动性危机"下的表现是否跟前两种类型
一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

START = "2020-01-15"
END = "2020-08-01"


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    print(f"拉取 GLD/IEF/SPY {START}~{END}(2020年新冠崩盘+早期反弹)数据...")
    data = yf.download(["GLD", "IEF", "SPY"], start=START, end=END, progress=False,
                       group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in ["GLD", "IEF", "SPY"]}
    price_df = pd.DataFrame(closes).dropna()
    rets = price_df.pct_change().dropna()

    print(f"数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}, {len(price_df)}天")
    corr = rets.corr()
    print(f"整个窗口: GLD跟SPY相关系数={corr.loc['GLD','SPY']:.4f}, IEF跟SPY相关系数={corr.loc['IEF','SPY']:.4f}")

    # 专门看崩盘最急的那几周(2020-02-19最高点 ~ 2020-03-23最低点)
    crash_start, crash_end = "2020-02-19", "2020-03-23"
    crash = price_df.loc[crash_start:crash_end]
    crash_rets = crash.pct_change().dropna()
    crash_corr = crash_rets.corr()
    print(f"\n真正崩盘期({crash_start}~{crash_end}, {len(crash)}天，SPY这段跌了多少见下面):")
    for t in ["SPY", "GLD", "IEF"]:
        total_ret = crash[t].iloc[-1] / crash[t].iloc[0] - 1
        print(f"  {t}: 区间总回报={total_ret*100:.2f}%")
    print(f"  崩盘期内GLD跟SPY相关系数={crash_corr.loc['GLD','SPY']:.4f}, IEF跟SPY相关系数={crash_corr.loc['IEF','SPY']:.4f}")

    for w in [0.2, 0.3]:
        blend_ret_gold = (1 - w) * rets["SPY"] + w * rets["GLD"]
        blend_nav_gold = (1 + blend_ret_gold).cumprod()
        blend_ret_bond = (1 - w) * rets["SPY"] + w * rets["IEF"]
        blend_nav_bond = (1 + blend_ret_bond).cumprod()
        spy_nav = (1 + rets["SPY"]).cumprod()
        print(f"\n  权重{w:.0%}: 纯SPY MaxDD={true_mdd(spy_nav.values)*100:.2f}%  "
              f"SPY+{w:.0%}黄金 MaxDD={true_mdd(blend_nav_gold.values)*100:.2f}%  "
              f"SPY+{w:.0%}IEF MaxDD={true_mdd(blend_nav_bond.values)*100:.2f}%")


if __name__ == "__main__":
    main()
