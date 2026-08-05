"""
Cycle39大修大补①(框架改动，新资产类别——货币)：美元指数(UUP ETF，
Invesco DB US Dollar Index Bullish Fund)作为第六个跨资产分散化候选，
沿用第84/85/165/171/195条同样的方法论(相关系数+四段区间blend对比)。
货币是这次会话完全没有测试过的资产类别——机制上美元走强通常伴随
risk-off环境(全球资金回流美元避险资产)，理论上可能跟SPY负相关，
但也可能因为强美元本身压制美股跨国企业海外收入而产生复杂的交互
效应，方向不预判，直接实测。
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
    print("拉取 UUP/GLD/IEF/SPY 数据...")
    tickers = ["UUP", "GLD", "IEF", "SPY"]
    data = yf.download(tickers, start="2009-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)
    print(f"UUP数据起点: {closes['UUP'].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  UUP-SPY相关系数: {rets['UUP'].corr(rets['SPY']):.4f}  "
              f"UUP-GLD相关系数: {rets['UUP'].corr(rets['GLD']):.4f}  "
              f"UUP-IEF相关系数: {rets['UUP'].corr(rets['IEF']):.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["UUP"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}UUP: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        w = 0.20
        blend_gld = (1 - w) * rets["SPY"] + w * rets["GLD"]
        print(f"  对照 SPY+20%GLD: Sharpe={sharpe_of(blend_gld):.3f}, MaxDD={true_mdd((1+blend_gld).cumprod().values)*100:.2f}%")


if __name__ == "__main__":
    main()
