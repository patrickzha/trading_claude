"""
Cycle41大修大补②(框架改动，新资产类别——单一商品聚焦测试)：原油(USO)
单独测试，比第165条的综合商品指数(DBC，能源/金属/农产品混合)更聚焦。
第165条已经发现DBC整体是混合偏负的分散化候选，这次测试单一能源商品
是否有不同的相关性结构(原油价格受地缘政治/供需冲击驱动，理论上可能
跟股票市场的相关性结构跟综合商品指数不同)。
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
    print("拉取 USO/GLD/SPY 数据...")
    tickers = ["USO", "GLD", "SPY"]
    data = yf.download(tickers, start="2009-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)
    print(f"USO数据起点: {closes['USO'].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  USO-SPY相关系数: {rets['USO'].corr(rets['SPY']):.4f}  USO-GLD相关系数: {rets['USO'].corr(rets['GLD']):.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["USO"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}USO: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")


if __name__ == "__main__":
    main()
