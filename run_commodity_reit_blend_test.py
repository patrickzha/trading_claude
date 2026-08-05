"""
Cycle34小修小补(c)：商品(DBC，第165条，混合偏负)和REITs(VNQ，第171条，
更清楚地证伪)两个分散化候选单独测都不成立，这次快速检验两者混合使用
(50/50各半)是不是有互补效应——如果DBC和VNQ本身低相关、亏损时点不同步，
混合后波动率可能被部分抵消，产生比任何单一候选更好的组合(这种"负负得正"
在分散化领域不是没有先例，第84/85条黄金+债券互补就是这类机制)。
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
    print("拉取 DBC/VNQ/SPY 数据...")
    tickers = ["DBC", "VNQ", "SPY"]
    data = yf.download(tickers, start="2006-06-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        print(f"  DBC-VNQ相关系数: {rets['DBC'].corr(rets['VNQ']):.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + (w / 2) * rets["DBC"] + (w / 2) * rets["VNQ"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}(DBC/VNQ各半): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        w = 0.20
        blend_dbc = (1 - w) * rets["SPY"] + w * rets["DBC"]
        blend_vnq = (1 - w) * rets["SPY"] + w * rets["VNQ"]
        print(f"  对照 SPY+20%纯DBC: Sharpe={sharpe_of(blend_dbc):.3f}, MaxDD={true_mdd((1+blend_dbc).cumprod().values)*100:.2f}%")
        print(f"  对照 SPY+20%纯VNQ: Sharpe={sharpe_of(blend_vnq):.3f}, MaxDD={true_mdd((1+blend_vnq).cumprod().values)*100:.2f}%")


if __name__ == "__main__":
    main()
