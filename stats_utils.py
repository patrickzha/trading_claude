"""
共享统计工具函数——这次会话`block_bootstrap_sharpe`被复制粘贴进了9个
独立测试脚本、`to_weekly`进了5个、`true_max_drawdown`类的函数进了9个。
这些历史脚本本身不需要回头重构(它们是已完成分析的存档，不是持续维护
的活代码，逐个改动风险大于收益)，但从现在开始，新脚本应该从这里
import，不再重新复制一遍。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_max_drawdown(nav_series) -> float:
    """peak-to-trough真实最大回撤，输入可以是list/array/Series。"""
    nav = np.asarray(nav_series, dtype=float)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def block_bootstrap_sharpe(rets, block: int = 4, n_boot: int = 3000,
                           periods_per_year: float = 52.0, seed: int = 42):
    """滑块自助法(block bootstrap)检验Sharpe显著性，块长默认4(周收益的
    自相关跨度)。返回(观测Sharpe, P(bootstrap Sharpe<=0))。"""
    rng = np.random.default_rng(seed)
    rets = np.asarray(rets)
    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([rets[i:i + block] for i in idx])[:n]
        boot.append(sample.mean() / (sample.std() + 1e-9) * np.sqrt(periods_per_year))
    boot = np.array(boot)
    obs = rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)
    return obs, float(np.mean(boot <= 0))


def to_weekly(daily_nav, price_df, get_weekly_trading_dates=None):
    """把逐日净值([(date, nav), ...]或Series)转成按周(周一到周五)的收益率
    序列。get_weekly_trading_dates不传时惰性从backtest.py导入，避免这个
    工具模块强依赖训练相关的重模块。"""
    if get_weekly_trading_dates is None:
        from backtest import get_weekly_trading_dates
    if isinstance(daily_nav, list):
        nav_series = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    else:
        nav_series = daily_nav
    week_pairs = get_weekly_trading_dates(price_df.index)
    rows = []
    for monday, friday in week_pairs:
        try:
            a, b = nav_series.loc[:monday].iloc[-1], nav_series.loc[:friday].iloc[-1]
        except IndexError:
            continue
        rows.append({"date": monday, "ret": b / a - 1})
    return pd.DataFrame(rows).set_index("date")["ret"]
