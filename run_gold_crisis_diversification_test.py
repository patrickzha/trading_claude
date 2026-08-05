"""
Cycle18大修大补①：黄金(GLD)分散化补测——README执行摘要里第69条附近明确
标注过黄金"有意思但未充分验证"：只在2022-2026这段测过(CAGR 14.77%远超TLT)，
但没有在真正的系统性危机区间(2004-2009，含2008年信用危机)测过，不能跟
已经过两次危机压力测试的债券方案(第65/69条)相提并论。这次补上2004-2009
这段关键危机数据，同时对比GLD跟债券(IEF)在四段区间的相关性/回撤保护
效果，尤其看2008年"黄金到底是不是真正的危机对冲工具"这个金融学里本身
就有争议的问题(不同于债券的"逃向安全资产"机制，黄金历史上在2008年初期
其实也随大宗商品一起下跌，直到危机后期央行放水阶段才走出独立行情)。

数据限制如实说明：GLD ETF 2004年11月才上市，2000-2002互联网泡沫那段
完全没有数据，四段区间里只有2004-2009这一段是真正的系统性危机测试，
这是黄金相对债券(TLT从1990s就有数据)的一个真实数据劣势，不是选择性
只测容易测的区间。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = {
    "2004-2009(含2008危机)": ("2004-12-01", "2009-06-02"),
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}

GOLD_WEIGHTS = [0.10, 0.20, 0.30]


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def sharpe_of(rets, periods_per_year=252):
    if len(rets) < 5:
        return float("nan")
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))


def main():
    print("拉取 GLD/IEF/SPY 2004-2026全区间数据...")
    data = yf.download(["GLD", "IEF", "SPY"], start="2004-01-01", end="2026-08-01",
                       progress=False, group_by="ticker", auto_adjust=True)
    closes = {}
    for t in ["GLD", "IEF", "SPY"]:
        df_t = data[t].dropna()
        closes[t] = df_t["Close"]
    price_df = pd.DataFrame(closes)
    print(f"数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足(GLD 2004年11月才上市)，跳过")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")

        corr = rets.corr()
        print(f"  GLD跟SPY相关系数: {corr.loc['GLD','SPY']:.4f}   IEF跟SPY相关系数: {corr.loc['IEF','SPY']:.4f}")

        spy_nav = sub["SPY"] / sub["SPY"].iloc[0]
        n_years = len(spy_nav) / 252
        spy_cagr = float(spy_nav.iloc[-1] ** (1 / n_years) - 1)
        spy_mdd = true_mdd(spy_nav.values)
        print(f"  纯SPY: CAGR={spy_cagr*100:.2f}%, MaxDD={spy_mdd*100:.2f}%, Sharpe={sharpe_of(rets['SPY']):.3f}")

        for w in GOLD_WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["GLD"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}黄金: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        # 对照：同样比例换成IEF(已验证的债券方案)
        for w in GOLD_WEIGHTS:
            blend_ret = (1 - w) * rets["SPY"] + w * rets["IEF"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            blend_mdd = true_mdd(blend_nav.values)
            print(f"  SPY+{w:.0%}IEF(对照): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")

        # 黄金+债券各半 vs 纯债券 vs 纯黄金(20%总敞口)
        w = 0.20
        blend_ret = (1 - w) * rets["SPY"] + (w / 2) * rets["GLD"] + (w / 2) * rets["IEF"]
        blend_nav = (1 + blend_ret).cumprod()
        blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
        blend_mdd = true_mdd(blend_nav.values)
        print(f"  SPY+10%黄金+10%IEF(混合): CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%, Sharpe={sharpe_of(blend_ret):.3f}")


if __name__ == "__main__":
    main()
