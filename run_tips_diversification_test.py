"""
Cycle37大修大补②(框架改动，新债券子类别——通胀风险管理)：TIPS(通胀
保值债券，用TIP ETF)代替名义国债(IEF/TLT)作为债券腿——现有债券对冲
(第65/66条)管理的是"名义利率风险"(加息周期债券价格下跌)，TIPS的
本金会跟随CPI指数调整，理论上应该在通胀意外走高的环境下(比如2021-2022
那波通胀)提供比名义债券更好的保护，这是第一次测试"通胀风险管理"这个
跟"名义利率风险管理"不同的机制维度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026(含2021-2022通胀高峰)": ("2021-08-02", "2026-07-30"),
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
    print("拉取 TIP/IEF/SPY 数据...")
    tickers = ["TIP", "IEF", "SPY"]
    data = yf.download(tickers, start="2010-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)
    print(f"TIP数据起点: {closes['TIP'].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  TIP-SPY相关系数: {rets['TIP'].corr(rets['SPY']):.4f}  IEF-SPY相关系数: {rets['IEF'].corr(rets['SPY']):.4f}")
        print(f"  TIP-IEF相关系数: {rets['TIP'].corr(rets['IEF']):.4f}(检验两者是否冗余)")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_tip = (1 - w) * rets["SPY"] + w * rets["TIP"]
            blend_ief = (1 - w) * rets["SPY"] + w * rets["IEF"]
            tip_nav = (1 + blend_tip).cumprod()
            ief_nav = (1 + blend_ief).cumprod()
            tip_cagr = float(tip_nav.iloc[-1] ** (1 / n_years) - 1)
            ief_cagr = float(ief_nav.iloc[-1] ** (1 / n_years) - 1)
            print(f"  SPY+{w:.0%}TIP: CAGR={tip_cagr*100:.2f}%, MaxDD={true_mdd(tip_nav.values)*100:.2f}%, Sharpe={sharpe_of(blend_tip):.3f}"
                  f"   |  SPY+{w:.0%}IEF(对照): CAGR={ief_cagr*100:.2f}%, MaxDD={true_mdd(ief_nav.values)*100:.2f}%, Sharpe={sharpe_of(blend_ief):.3f}")


if __name__ == "__main__":
    main()
