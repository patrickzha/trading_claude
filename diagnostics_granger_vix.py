"""
诊断脚本：VIX 是否真的领先于股票池的未来收益？
=============================================
check_macro_warning() 里的核心假设是"VIX大涨/大盘破位 -> 未来该规避风险"，
但这条规则的阈值（VIX 5日涨30%、动量回撤这类数字）是拍脑袋定的，没有做过
因果检验。这里用 Granger 因果检验看历史上 VIX 的变化是不是真的领先于这个
股票池的未来收益，而不是想当然地认为"VIX涨=危险"。

纯诊断用途，不进主流程，跑一次看结论即可；结论用来决定 HMM regime 的设计
要不要继续依赖 VIX 这一个变量。
"""
from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from backtest import fetch_market_data


def _adf_report(series: pd.Series, name: str) -> None:
    series = series.dropna()
    stat, pvalue = adfuller(series)[:2]
    verdict = "平稳" if pvalue < 0.05 else "非平稳（后面的检验结果要谨慎看）"
    print(f"  ADF检验 {name}: p={pvalue:.4f} -> {verdict}")


def run_granger_test(max_lag: int = 10) -> None:
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()

    # 股票池横截面平均日收益率，近似"这个股票池的大盘"
    market_ret = c_df.mean(axis=1).pct_change().dropna()
    vix_chg = vix.pct_change().dropna()

    common_idx = market_ret.index.intersection(vix_chg.index)
    market_ret = market_ret.loc[common_idx]
    vix_chg = vix_chg.loc[common_idx]

    print("=" * 70)
    print(" 【Granger 因果检验：VIX变化 是否领先于 股票池未来收益】")
    print("=" * 70)
    _adf_report(market_ret, "股票池日收益率")
    _adf_report(vix_chg, "VIX日变化率")

    data = pd.DataFrame({"market_ret": market_ret, "vix_chg": vix_chg}).dropna()

    print(f"\n 检验假设: VIX变化是否 Granger-cause 股票池未来收益（滞后1~{max_lag}天）")
    print("-" * 70)
    # statsmodels 约定：第二列是否 Granger-cause 第一列
    results = grangercausalitytests(data[["market_ret", "vix_chg"]], maxlag=max_lag, verbose=False)

    significant_lags = []
    for lag in range(1, max_lag + 1):
        pvalue = results[lag][0]["ssr_ftest"][1]
        marker = "  <- 显著 (p<0.05)" if pvalue < 0.05 else ""
        print(f"  滞后{lag:2d}天: p={pvalue:.4f}{marker}")
        if pvalue < 0.05:
            significant_lags.append(lag)

    print("-" * 70)
    if significant_lags:
        print(f" 结论: 在滞后 {significant_lags} 天上检测到显著的领先关系，"
              f"VIX变化对该股票池未来收益有一定预测力，现有VIX熔断规则有统计支持。")
    else:
        print(f" 结论: 在1~{max_lag}天的滞后阶数里都没有检测到显著的领先关系。"
              f"现有VIX熔断规则（vix_curr>25、5日涨30%这类阈值）目前没有统计证据"
              f"支持，很可能只是在这段回测历史上凑巧生效，不建议当成可靠规律。"
              f"HMM regime 设计时不应该只依赖 VIX 这一个变量，应该加大盘自身的"
              f"动量/波动率信息一起判断状态。")
    print("=" * 70)


if __name__ == "__main__":
    run_granger_test()
