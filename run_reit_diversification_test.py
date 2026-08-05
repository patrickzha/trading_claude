"""
Cycle33大修大补②(框架改动，新资产类别)：REITs房地产信托(VNQ，Vanguard Real
Estate ETF)作为第四条跨资产分散化候选——第65条那句"可能需要完全不同的
资产类别（大宗商品、债券、另类资产）"里，商品(第165条)已经测过并证伪
(实质冗余、牛市里拖累)，这次测REITs这个同样常见的另类分散化候选，机制
上跟商品/黄金/债券都不同：REITs本质是"股票化的地产"，兼具股票的成长性
和债券的利率敏感性(REITs估值对利率变化很敏感，跟债券类似)，理论上跟
纯股票的相关性应该介于股票和债券之间，能不能提供有意义的分散化是这次
要检验的问题。

方法论跟第84/85条(黄金)、第165条(商品)完全一致：四段区间相关系数+
blend CAGR/MaxDD/Sharpe对比，同时跟GLD/IEF做交叉相关性矩阵检验冗余。

数据限制如实说明：VNQ 2004年9月上市，比GLD(2004年11月)略早，2004-2009
这段危机窗口覆盖相对完整（只缺2004年9-12月约3个月）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2004-2009(含2008危机)": ("2004-12-01", "2009-06-02"),
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}

WEIGHTS = [0.10, 0.20, 0.30]


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def sharpe_of(rets, periods_per_year=252):
    if len(rets) < 5:
        return float("nan")
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))


def main():
    print("拉取 VNQ/GLD/IEF/SPY 2004-2026全区间数据...")
    tickers = ["VNQ", "GLD", "IEF", "SPY"]
    data = yf.download(tickers, start="2004-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {}
    for t in tickers:
        df_t = data[t].dropna()
        closes[t] = df_t["Close"]
    price_df = pd.DataFrame(closes)
    print(f"VNQ数据起点: {closes['VNQ'].index[0].date()}")
    print(f"整体数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}\n")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过(n={len(sub)}天)")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天，实际起点{sub.index[0].date()}) ===")

        corr = rets.corr()
        print(f"  两两相关系数: VNQ-SPY={corr.loc['VNQ','SPY']:.4f}  "
              f"VNQ-GLD={corr.loc['VNQ','GLD']:.4f}  VNQ-IEF={corr.loc['VNQ','IEF']:.4f}  "
              f"GLD-SPY={corr.loc['GLD','SPY']:.4f}  IEF-SPY={corr.loc['IEF','SPY']:.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["VNQ"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}REITs(VNQ): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        for ref_ticker in ["GLD", "IEF"]:
            w = 0.20
            blend_ret = (1 - w) * rets["SPY"] + w * rets[ref_ticker]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+20%{ref_ticker}(对照): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")


if __name__ == "__main__":
    main()
