"""
Cycle35大修大补①(框架改动，新数据维度)：国债收益率曲线倒挂(10年期^TNX
减3个月期^IRX利差)是否领先于股票池未来收益——这次会话此前测过的宏观
信号全部集中在波动率维度(现货VIX第1条、VIX期限结构第175/176条)，
收益率曲线是完全不同的数据维度(利率预期，不是波动率预期)，经济学
文献里"10年-3个月利差倒挂"是最经典、最被广泛引用的衰退先行指标之一
(纽约联储的衰退概率模型核心就是这个利差)。跟第175条同样的方法论：
先做Granger可行性检验，通过再投入策略集成测试。
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from backtest import fetch_market_data


def _adf_report(series: pd.Series, name: str) -> None:
    series = series.dropna()
    stat, pvalue = adfuller(series)[:2]
    verdict = "平稳" if pvalue < 0.05 else "非平稳(后面的检验结果要谨慎看)"
    print(f"  ADF检验 {name}: p={pvalue:.4f} -> {verdict}")


def run_granger_test(signal: pd.Series, signal_name: str, market_ret: pd.Series, max_lag: int = 10):
    common_idx = market_ret.index.intersection(signal.index)
    market_ret_aligned = market_ret.loc[common_idx]
    signal_aligned = signal.loc[common_idx]

    print(f"\n{'='*70}\n 【Granger因果检验：{signal_name} 是否领先于 股票池未来收益】(n={len(common_idx)})\n{'='*70}")
    _adf_report(market_ret_aligned, "股票池日收益率")
    _adf_report(signal_aligned, signal_name)

    data = pd.DataFrame({"market_ret": market_ret_aligned, "signal": signal_aligned}).dropna()
    if len(data) < 100:
        print(f"  样本不足(n={len(data)})，跳过")
        return []

    results = grangercausalitytests(data[["market_ret", "signal"]], maxlag=max_lag, verbose=False)
    significant_lags = []
    for lag in range(1, max_lag + 1):
        pvalue = results[lag][0]["ssr_ftest"][1]
        marker = "  <- 显著(p<0.05)" if pvalue < 0.05 else ""
        print(f"  滞后{lag:2d}天: p={pvalue:.4f}{marker}")
        if pvalue < 0.05:
            significant_lags.append(lag)

    if significant_lags:
        print(f" 结论: 在滞后{significant_lags}天上检测到显著领先关系。")
    else:
        print(f" 结论: 1~{max_lag}天滞后阶数都没有检测到显著领先关系。")
    return significant_lags


def main():
    print("拉取股票池行情+10年期/3个月期国债收益率数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()
    market_ret = c_df.mean(axis=1).pct_change().dropna()

    rate_data = yf.download(["^TNX", "^IRX"], start=c_df.index[0], end=c_df.index[-1],
                             progress=False, group_by="ticker", auto_adjust=True)
    tnx = rate_data["^TNX"]["Close"].dropna()
    irx = rate_data["^IRX"]["Close"].dropna()

    common = tnx.index.intersection(irx.index)
    spread = (tnx.loc[common] - irx.loc[common])
    spread_chg = spread.diff().dropna()
    inverted_flag = (spread < 0).astype(float)
    inverted_chg = inverted_flag.diff().dropna()

    print(f"数据覆盖: {common[0].date()} ~ {common[-1].date()}, n={len(common)}天")
    print(f"倒挂(10Y<3M)出现比例: {inverted_flag.mean()*100:.1f}%")
    print(f"当前利差(最新): {spread.iloc[-1]:.2f}")

    sig_spread = run_granger_test(spread_chg, "10Y-3M利差变化", market_ret)
    sig_flag = run_granger_test(inverted_chg, "倒挂状态切换(0/1)", market_ret)

    print(f"\n{'='*70}\n 汇总\n{'='*70}")
    print(f" 利差连续变化: {'有显著领先关系' if sig_spread else '无显著领先关系'}")
    print(f" 倒挂状态切换: {'有显著领先关系' if sig_flag else '无显著领先关系'}")
    if not sig_spread and not sig_flag:
        print(" 结论: 收益率曲线信号(连续利差+倒挂状态切换)都没有通过Granger检验，"
              "不建议投入后续的策略集成工程量。")
    else:
        print(" 结论: 至少一个信号通过了Granger检验，值得投入后续集成验证。")


if __name__ == "__main__":
    main()
