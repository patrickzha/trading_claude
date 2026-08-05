"""
Cycle42大修大补①(框架改动，新资产类别——贵金属细分)：白银(SLV)分散化
测试——黄金(第84/85条)是这次会话证据最扎实的分散化发现之一，但白银
虽然同属贵金属，工业属性(大量用于电子元件、太阳能面板生产)比黄金
强得多，理论上白银的价格驱动因素可能更接近工业商品(第165/220条已
证伪的方向)而不是纯粹的避险资产(黄金)，这次测试哪种机制主导白银
的实际相关性结构。
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
    print("拉取 SLV/GLD/USO/SPY 数据...")
    tickers = ["SLV", "GLD", "USO", "SPY"]
    data = yf.download(tickers, start="2009-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)
    print(f"SLV数据起点: {closes['SLV'].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  SLV-SPY相关系数: {rets['SLV'].corr(rets['SPY']):.4f}  "
              f"SLV-GLD相关系数: {rets['SLV'].corr(rets['GLD']):.4f}  "
              f"SLV-USO相关系数: {rets['SLV'].corr(rets['USO']):.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["SLV"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}SLV: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        w = 0.20
        blend_gld = (1 - w) * rets["SPY"] + w * rets["GLD"]
        print(f"  对照 SPY+20%GLD: Sharpe={sharpe_of(blend_gld):.3f}, MaxDD={true_mdd((1+blend_gld).cumprod().values)*100:.2f}%")


if __name__ == "__main__":
    main()
