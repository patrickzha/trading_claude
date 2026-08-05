"""
Cycle66大修大补②(框架改动，新模型架构——条件波动率预测)：低波动腿候选排序
从"trailing 60日已实现波动率"(简单历史统计量，见`low_vol_investing.py`)
换成GARCH(1,1)一步向前预测的条件波动率——这次会话此前测过的模型架构类
候选(HMM regime、LSTM、stacking元模型)全部是在动量腿的收益预测上尝试，
从没有把时间序列波动率预测模型用在低波动腿的"波动率排序"这个环节，是
一个没试过的具体应用点，即使GARCH本身不是全新算法。

理论依据：已实现波动率(简单历史标准差)对波动率的"聚集性"(vol clustering，
高波动之后倾向于继续高波动)反应滞后，GARCH(1,1)显式建模条件方差
sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2，理论上能
更快地对近期波动冲击做出反应，用来选"未来波动率最低"的股票，可能比
简单看过去60天的标准差更准。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def select_garch_low_vol_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                                     top_n: int = 30, lookback_days: int = 250):
    from arch import arch_model
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna().tail(lookback_days)
        if len(px) < 120:
            continue
        rets = px.pct_change().dropna() * 100  # arch包在百分比尺度上数值更稳定
        if len(rets) < 100:
            continue
        try:
            am = arch_model(rets, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = float(np.sqrt(forecast.variance.values[-1, 0])) / 100.0
            rows.append({"ticker": t, "garch_vol": cond_vol_forecast})
        except Exception:
            continue

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]
    df = pd.DataFrame(rows).sort_values("garch_vol")
    return df["ticker"].head(top_n).tolist()
