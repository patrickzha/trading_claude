"""
Cycle12大修大补：真正跳出股票这个资产类别——第64条已经证明换行业/市值/
国家分散都逃不开股票资产类别本身的高相关性，尤其在危机时刻。这次测
债券(TLT长期国债/IEF中期国债)和黄金(GLD)这两个经典的"股债/避险"分散
来源，跟第62条美股内部因子组合、第64条国际轮动的相关性对比，尤其看
2008年危机区间——债券在危机期间"逃向安全资产"是金融学里最经典的
分散化机制，理论上应该比任何股票内部的分散化都更有效。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

# TLT(20年期以上国债ETF)历史只到2002年，IEF(7-10年期国债)也类似，GLD(黄金)2004年上市
ASSETS = {"TLT": "20年期以上美国国债ETF", "IEF": "7-10年期美国国债ETF", "GLD": "黄金ETF"}

PERIODS = {
    "2004-2009(含危机)": ("2004-06-01", "2009-06-02"),
    "2009-2014": ("2009-06-01", "2014-06-02"),
    "2014-2020": ("2014-06-02", "2020-12-31"),
    "2022-2026": ("2021-08-02", "2026-07-30"),
}


def _extract_close(df):
    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Close", axis=1, level=0)
    else:
        close = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
    return close.squeeze(axis=1)


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    print(f"拉取 {list(ASSETS.keys())} + SPY 2004-2026全区间数据...")
    data = yf.download(list(ASSETS.keys()) + ["SPY"], start="2004-01-01", end="2026-08-01",
                       progress=False, group_by="ticker", auto_adjust=True)
    closes = {}
    for t in list(ASSETS.keys()) + ["SPY"]:
        try:
            df_t = data[t].dropna()
            closes[t] = df_t["Close"]
        except Exception as e:
            print(f"  {t}拉取失败: {e}")
    price_df = pd.DataFrame(closes)
    print(f"数据范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}, 覆盖: {list(price_df.columns)}")

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 100:
            print(f"\n=== {label} ===\n  数据不足，跳过（可能是TLT/GLD这段还没上市）")
            continue
        rets = sub.pct_change().dropna()
        print(f"\n=== {label} (n={len(sub)}天) ===")
        corr_with_spy = rets.corr()["SPY"]
        for asset in ASSETS:
            if asset in corr_with_spy.index:
                print(f"  {asset}({ASSETS[asset]}) 跟SPY相关系数: {corr_with_spy[asset]:.4f}")

        for asset in list(ASSETS.keys()) + ["SPY"]:
            if asset not in sub.columns:
                continue
            nav = sub[asset] / sub[asset].iloc[0]
            n_years = len(nav) / 252
            cagr = float(nav.iloc[-1] ** (1 / n_years) - 1)
            mdd = true_mdd(nav.values)
            print(f"  {asset}: CAGR={cagr*100:.2f}%, MaxDD={mdd*100:.2f}%")

        # 60/40股债组合 vs 纯股票，看2008年这种危机期间债券有没有真的对冲
        if "TLT" in sub.columns:
            blend_ret = 0.6 * rets["SPY"] + 0.4 * rets["TLT"]
            blend_nav = (1 + blend_ret).cumprod()
            blend_mdd = true_mdd(blend_nav.values)
            n_years = len(blend_nav) / 252
            blend_cagr = float(blend_nav.iloc[-1] ** (1 / n_years) - 1)
            print(f"  60/40股债组合: CAGR={blend_cagr*100:.2f}%, MaxDD={blend_mdd*100:.2f}%（对照纯SPY回撤）")


if __name__ == "__main__":
    main()
