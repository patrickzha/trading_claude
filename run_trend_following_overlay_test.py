"""
Cycle40大修大补①(框架改动，新机制——时序趋势跟随资产配置)：用trend-
following(资产自己trailing 6个月收益是否为正)动态决定黄金/美元/债券
三条非股票分散化腿的持有与否，代替现有静态`gold_split`/`dollar_split`
固定比例分配。这是"时序动量"(time-series momentum，只看资产自己的
趋势方向，不做横截面排名)这个范式，是CTA/managed futures类策略的
理论基础，跟这次会话所有测试过的横截面方法(个股排名、板块排名)是
完全不同的机制。

设计：每月重新评估一次，某条腿trailing 126个交易日收益为正就持有
(按原定权重)，为负就把这部分权重转为现金(不做空)，代替静态"永远
按固定比例持有"。
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

TREND_WINDOW = 126
REBALANCE_DAYS = 21
NON_EQUITY_WEIGHT = 0.4  # 跟full_system.py的DEFAULT_BOND_WEIGHT一致


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def sharpe_of(rets, periods_per_year=252):
    if len(rets) < 5:
        return float("nan")
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))


def backtest_static(sub, spy_ret, gold_ret, dollar_ret, bond_ret, split=(1/3, 1/3, 1/3)):
    g, d, b = split
    blend = (1 - NON_EQUITY_WEIGHT) * spy_ret + NON_EQUITY_WEIGHT * (g * gold_ret + d * dollar_ret + b * bond_ret)
    return blend.dropna()


def backtest_trend_following(sub, spy_ret, gold_px, dollar_px, bond_px, spy_ret_full):
    index = sub.index
    n = len(index)
    each_w = NON_EQUITY_WEIGHT / 3
    daily_rets = []
    current_flags = {"gold": True, "dollar": True, "bond": True}

    for i in range(TREND_WINDOW, n):
        d = index[i]
        if (i - TREND_WINDOW) % REBALANCE_DAYS == 0:
            for name, px in [("gold", gold_px), ("dollar", dollar_px), ("bond", bond_px)]:
                window = px.loc[:d].tail(TREND_WINDOW + 1)
                if len(window) >= TREND_WINDOW:
                    trend_ret = window.iloc[-1] / window.iloc[0] - 1
                    current_flags[name] = trend_ret > 0

        r_spy = spy_ret_full.get(d, np.nan)
        r_gold = gold_px.pct_change().get(d, np.nan) if current_flags["gold"] else 0.0
        r_dollar = dollar_px.pct_change().get(d, np.nan) if current_flags["dollar"] else 0.0
        r_bond = bond_px.pct_change().get(d, np.nan) if current_flags["bond"] else 0.0
        if pd.isna(r_spy):
            continue
        r_gold = r_gold if pd.notna(r_gold) else 0.0
        r_dollar = r_dollar if pd.notna(r_dollar) else 0.0
        r_bond = r_bond if pd.notna(r_bond) else 0.0
        blend_ret = (1 - NON_EQUITY_WEIGHT) * r_spy + each_w * (r_gold + r_dollar + r_bond)
        daily_rets.append((d, blend_ret))

    return daily_rets


def metrics_from_rets(rets):
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    n_years = len(rets) / 252
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = sharpe_of(rets)
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


def main():
    print("拉取 GLD/UUP/IEF/SPY 数据...")
    tickers = ["GLD", "UUP", "IEF", "SPY"]
    data = yf.download(tickers, start="2009-01-01", end="2026-08-01",
                        progress=False, group_by="ticker", auto_adjust=True)
    closes = {t: data[t]["Close"].dropna() for t in tickers}
    price_df = pd.DataFrame(closes)

    for label, (start, end) in PERIODS.items():
        sub = price_df.loc[start:end].dropna()
        if len(sub) < 200:
            print(f"\n=== {label} ===\n  数据不足，跳过")
            continue
        rets = sub.pct_change().dropna()
        spy_ret_full = rets["SPY"]

        static_blend = backtest_static(sub, rets["SPY"], rets["GLD"], rets["UUP"], rets["IEF"])
        s_cagr, s_sharpe, s_mdd = metrics_from_rets(static_blend.values)

        tf_rets_list = backtest_trend_following(sub, rets["SPY"], sub["GLD"], sub["UUP"], sub["IEF"], spy_ret_full)
        tf_rets = [r for _, r in tf_rets_list]
        if len(tf_rets) < 30:
            print(f"\n=== {label} ===\n  趋势跟随样本不足，跳过")
            continue
        t_cagr, t_sharpe, t_mdd = metrics_from_rets(tf_rets)

        print(f"\n=== {label} ===")
        print(f"  静态三分(各1/3): CAGR={s_cagr*100:.2f}%, Sharpe={s_sharpe:.3f}, MaxDD={s_mdd*100:.2f}%")
        print(f"  趋势跟随动态开关: CAGR={t_cagr*100:.2f}%, Sharpe={t_sharpe:.3f}, MaxDD={t_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
