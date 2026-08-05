"""
Cycle32大修大补①(框架改动，新资产类别)：大宗商品(DBC，Invesco DB Commodity
Index Tracking Fund)作为第三条跨资产分散化腿——第65条明确指出"如果想要
真正独立于美股的分散化...可能需要完全不同的资产类别（大宗商品、债券、
另类资产）"，国际股票轮动(第65条)已经被证伪(危机期间高相关、跌得更深)，
这次直接测试大宗商品这个第65条点名但没有实测过的候选。

跟第84/85条黄金分散化测试用完全相同的方法论(相关系数+四段区间blend
CAGR/MaxDD/Sharpe对比)，同时把GLD/IEF也拉进来做三者两两相关性矩阵，
检验商品跟已经验证过的黄金/债券这两条腿是否有实质冗余(如果商品价格
本质是"另一种通胀对冲"，可能跟黄金高相关，冗余没有额外分散化价值)。

数据限制如实说明：DBC 2006年2月才上市，比GLD(2004年11月)更晚，
2004-2009这段危机窗口只能覆盖2006年2月之后约3年，不是完整覆盖，
如实标注，不回避。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "2004-2009(含2008危机，DBC数据从2006-02开始，覆盖不完整)": ("2004-12-01", "2009-06-02"),
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
    print("拉取 DBC/GLD/IEF/SPY 2004-2026全区间数据...")
    tickers = ["DBC", "GLD", "IEF", "SPY"]
    data = yf.download(tickers, start="2004-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {}
    for t in tickers:
        df_t = data[t].dropna()
        closes[t] = df_t["Close"]
    price_df = pd.DataFrame(closes)
    print(f"DBC数据起点: {closes['DBC'].index[0].date()}")
    print(f"整体数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}\n")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过(n={len(sub)}天)")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天，实际起点{sub.index[0].date()}) ===")

        corr = rets.corr()
        print(f"  两两相关系数: DBC-SPY={corr.loc['DBC','SPY']:.4f}  "
              f"DBC-GLD={corr.loc['DBC','GLD']:.4f}  DBC-IEF={corr.loc['DBC','IEF']:.4f}  "
              f"GLD-SPY={corr.loc['GLD','SPY']:.4f}  IEF-SPY={corr.loc['IEF','SPY']:.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["DBC"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}商品(DBC): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        # 对照：同样比例换成GLD/IEF(已验证方案)
        for ref_ticker in ["GLD", "IEF"]:
            w = 0.20
            blend_ret = (1 - w) * rets["SPY"] + w * rets[ref_ticker]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+20%{ref_ticker}(对照): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        # 商品+黄金+债券三者等分(共20%敞口) vs 纯商品20%
        w = 0.20
        blend_ret = (1 - w) * rets["SPY"] + (w / 3) * rets["DBC"] + (w / 3) * rets["GLD"] + (w / 3) * rets["IEF"]
        blend_nav = (1 + blend_ret).cumprod()
        blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
        blend_mdd = true_mdd(blend_nav.values)
        print(f"  SPY+商品/黄金/债券各6.67%(三者混合): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")


if __name__ == "__main__":
    main()
