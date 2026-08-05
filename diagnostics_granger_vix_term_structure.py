"""
Cycle34大修大补①(框架改动，新数据维度)的前置可行性检验：VIX期限结构
(VIX9D/VIX/VIX3M三者关系，反映短期vs中期波动率预期的相对水平，
contango/backwardation结构)是否比现货VIX水平(第1条已经证明没有格兰杰
因果关系)更能预测这个股票池的未来收益。这是完整大修大补投入之前的
feasibility gate，如实做诊断——如果期限结构信号同样没有因果支持，
及时止损，不投入后续的策略集成工程量，直接如实记录负面结果。

VIX9D(9天期隐含波动率)/VIX(30天)/VIX3M(3个月期)三者的相对关系比单纯
现货VIX多一个维度：正常市场(contango，VIX9D<VIX<VIX3M，波动率期限
结构向上倾斜)vs恐慌市场(backwardation，短期恐慌情绪推高VIX9D超过
VIX3M)，理论上backwardation是比"VIX绝对水平"更精确的恐慌信号。
"""
from __future__ import annotations

import numpy as np
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
    print("拉取股票池行情+VIX9D/VIX3M期限结构数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()
    market_ret = c_df.mean(axis=1).pct_change().dropna()

    term_data = yf.download(["^VIX9D", "^VIX3M"], start=c_df.index[0], end=c_df.index[-1],
                             progress=False, group_by="ticker", auto_adjust=True)
    vix9d = term_data["^VIX9D"]["Close"].dropna()
    vix3m = term_data["^VIX3M"]["Close"].dropna()

    common = vix.index.intersection(vix9d.index).intersection(vix3m.index)
    vix_c, vix9d_c, vix3m_c = vix.loc[common], vix9d.loc[common], vix3m.loc[common]

    term_slope = (vix3m_c - vix9d_c) / vix_c
    term_slope_chg = term_slope.diff().dropna()
    backwardation_flag = (vix9d_c > vix3m_c).astype(float)
    backwardation_chg = backwardation_flag.diff().dropna()

    print(f"数据覆盖: {common[0].date()} ~ {common[-1].date()}, n={len(common)}天")
    print(f"backwardation(短期恐慌)出现比例: {backwardation_flag.mean()*100:.1f}%")

    sig_slope = run_granger_test(term_slope_chg, "VIX期限结构斜率变化((VIX3M-VIX9D)/VIX)", market_ret)
    sig_flag = run_granger_test(backwardation_chg, "backwardation状态切换(0/1)", market_ret)

    print(f"\n{'='*70}\n 汇总\n{'='*70}")
    print(f" 期限结构斜率变化: {'有显著领先关系' if sig_slope else '无显著领先关系'}")
    print(f" backwardation状态切换: {'有显著领先关系' if sig_flag else '无显著领先关系'}")
    if not sig_slope and not sig_flag:
        print(" 结论: VIX期限结构信号(斜率+backwardation切换)都没有通过Granger检验，"
              "跟第1条现货VIX水平一样，没有统计上的领先关系支持，不建议投入后续的"
              "熔断规则集成工程量。")
    else:
        print(" 结论: 至少一个期限结构信号通过了Granger检验，值得投入后续集成验证。")


if __name__ == "__main__":
    main()
