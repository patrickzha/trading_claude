"""
Cycle39大修大补②(框架改动，新数据维度——波动率风险溢价)的前置可行性
检验：VRP(variance risk premium，隐含波动率VIX减去trailing已实现波动率
的价差)是否领先于股票池未来收益。不同于第1条(现货VIX水平)、第175/176条
(VIX期限结构)，这次测的是"隐含vs已实现"这个价差本身——波动率交易文献
里VRP长期为正(隐含波动率系统性高于已实现波动率，是"卖保险"一方长期
收取的风险溢价)，VRP的变化(尤其是VRP骤降甚至转负)有时被认为预示市场
应力上升。跟第175/182条同样的方法论：先做Granger可行性检验，通过再
投入策略集成测试。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
    print("拉取股票池行情+VIX数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()
    market_ret = c_df.mean(axis=1).pct_change().dropna()

    realized_vol = market_ret.rolling(21).std() * np.sqrt(252) * 100  # 年化realized vol，跟VIX同单位(百分比点)
    vrp = (vix - realized_vol).dropna()
    vrp_chg = vrp.diff().dropna()

    print(f"VRP均值: {vrp.mean():.2f}, VRP为负(隐含<已实现)的比例: {(vrp < 0).mean()*100:.1f}%")

    sig = run_granger_test(vrp_chg, "VRP变化(VIX-trailing21日已实现波动率)", market_ret)

    print(f"\n{'='*70}\n 汇总\n{'='*70}")
    if not sig:
        print(" 结论: VRP变化没有通过Granger检验，不建议投入后续的策略集成工程量。")
    else:
        print(" 结论: VRP变化通过了Granger检验，值得投入后续集成验证。")


if __name__ == "__main__":
    main()
