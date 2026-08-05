"""
Cycle37大修大补①(框架改动，新资产类别)：比特币(BTC-USD)作为第五个跨
资产分散化候选——沿用黄金(第84/85条)/商品(第165条)/REITs(第171条)
完全相同的方法论(相关系数+四段区间blend CAGR/MaxDD/Sharpe对比)。
比特币是这次会话测试过的资产类别里性质最特殊的一个：无现金流、无
内在价值锚定(不像股票有盈利、债券有利息、黄金有工业/首饰需求、
商品有实体消费)，估值逻辑完全不同，理论上跟传统资产的相关性可能
最低，但历史波动率也是这次测试过的所有候选里最高的，净分散化效果
需要实测，不能预判方向。

数据限制如实说明：BTC-USD在Yahoo Finance的历史数据从2014年9月才
开始，覆盖不了2004-2009危机段，这次只能测2009-2014(部分)、2014-2020、
2022-2026三段。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2014-2020(BTC数据从2014-09开始)": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}

WEIGHTS = [0.05, 0.10, 0.20]


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def sharpe_of(rets, periods_per_year=252):
    if len(rets) < 5:
        return float("nan")
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))


def main():
    print("拉取 BTC-USD/GLD/IEF/SPY 数据...")
    tickers = ["BTC-USD", "GLD", "IEF", "SPY"]
    data = yf.download(tickers, start="2014-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {}
    for t in tickers:
        df_t = data[t].dropna()
        closes[t] = df_t["Close"]
    price_df = pd.DataFrame(closes)
    print(f"BTC-USD数据起点: {closes['BTC-USD'].index[0].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过(n={len(sub)}天)")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天，实际起点{sub.index[0].date()}) ===")

        corr = rets.corr()
        print(f"  两两相关系数: BTC-SPY={corr.loc['BTC-USD','SPY']:.4f}  "
              f"BTC-GLD={corr.loc['BTC-USD','GLD']:.4f}  BTC-IEF={corr.loc['BTC-USD','IEF']:.4f}")
        print(f"  BTC年化波动率: {rets['BTC-USD'].std()*np.sqrt(252)*100:.1f}%  "
              f"(对照SPY: {rets['SPY'].std()*np.sqrt(252)*100:.1f}%)")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["BTC-USD"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}BTC: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        w = 0.20
        blend_gld = (1 - w) * rets["SPY"] + w * rets["GLD"]
        print(f"  对照 SPY+20%GLD: Sharpe={sharpe_of(blend_gld):.3f}, MaxDD={true_mdd((1+blend_gld).cumprod().values)*100:.2f}%")


if __name__ == "__main__":
    main()
