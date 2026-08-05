"""
Cycle61大修大补②(框架改动，新数据维度——尾部风险定价)的前置可行性检验：
CBOE SKEW指数(^SKEW)是否领先于股票池未来收益。SKEW指数衡量标普500深度
虚值看跌期权的隐含波动率相对平值期权的溢价程度，反映市场对"黑天鹅"式
尾部崩盘的定价程度——机制上跟第1条(VIX现货水平，衡量整体波动率预期)、
第175/176条(VIX期限结构，衡量近月vs远月波动率预期的相对关系)都不同，
这次测的是"崩盘尾部风险"这个更狭窄、更具体的维度，不是波动率整体水平。
跟这次会话所有宏观信号同样的方法论：先做Granger可行性检验，通过再决定
是否投入策略集成测试。
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
    print("拉取股票池行情数据...")
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()
    market_ret = c_df.mean(axis=1).pct_change().dropna()

    print("拉取CBOE SKEW指数历史数据...")
    skew_data = yf.download("^SKEW", start=c_df.index[0], end=c_df.index[-1], progress=False, auto_adjust=True)
    skew_close = skew_data["Close"]
    if isinstance(skew_close, pd.DataFrame):
        skew_close = skew_close.iloc[:, 0]
    skew_close = skew_close.dropna()
    print(f"  SKEW指数区间: {skew_close.index[0].date()} ~ {skew_close.index[-1].date()}, {len(skew_close)}天")

    skew_level = skew_close.reindex(c_df.index).ffill()
    skew_chg_5d = skew_level.diff(5)

    run_granger_test(skew_level.diff().dropna(), "SKEW指数日变化(水平)", market_ret)
    run_granger_test(skew_chg_5d, "SKEW指数5日变化", market_ret)


if __name__ == "__main__":
    main()
