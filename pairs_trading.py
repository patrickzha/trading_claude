"""
统计套利/配对交易(pairs trading)引擎——跟现有周频动量策略完全独立的
新框架，赌的是"两只历史上一起动的股票，价差偏离后会均值回归"这个
数学结构，不是"预测哪只股票会涨"。

方法（经典 distance method，Gatev/Goetzmann/Rouwenhorst 2006）：
1. 形成期(formation_days，默认252个交易日约1年)：同一行业内，把每只票的
   价格归一化成"从形成期起点开始的累计净值"（起点=1.0），选出归一化净值
   序列距离(误差平方和)最小的若干对——历史上走得最像的一对。
2. 交易期(trading_days，默认63个交易日约1季度)：继续跟踪这对票的净值差
   (spread = normA - normB)，用形成期内spread的标准差算z-score，
   |z-score|超过入场阈值(默认2)就开仓——spread过高(A比B涨多了)做空A/
   做多B，反过来同理；z-score回归到出场阈值(默认0.5)附近平仓；
   |z-score|继续发散超过止损阈值(默认4)提前止损；交易期结束强制平仓。
3. 往前滚动：交易期结束后，用最新的形成期重新选一批pair，重复，没有
   前视偏差（形成期和交易期都严格按时间顺序，不会用到未来数据选pair）。

必然含有做空腿——这是配对交易这个策略类型本身的定义（同时做多做空
才能锁定"相对价差"、不暴露在大盘方向性风险里），跟之前"允许对最差
候选做空"是完全不同的两件事：这次是用户在"统计套利"这个方向下明确
批准的，不是对现有动量策略加空头。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_pairs(price_df: pd.DataFrame, sector_map: dict, formation_end_idx: int,
                  formation_days: int = 252, top_n_per_sector: int = 2, min_history: int = 200,
                  restrict_to_sector: bool = True, top_n_global: int = 20):
    """在形成期窗口内选pair，按归一化价格距离(误差平方和)最小排序。
    restrict_to_sector=True(默认)时只在同行业内配对；False时按Gatev等
    (2006)原始distance method，不分行业、全市场找距离最小的top_n_global对。
    不用到形成期结束点之后的任何数据。"""
    start_idx = max(0, formation_end_idx - formation_days)
    window = price_df.iloc[start_idx:formation_end_idx]
    if len(window) < min_history:
        return []

    if restrict_to_sector:
        groups: dict[str, list[str]] = {}
        for t in window.columns:
            s = sector_map.get(t, "UNKNOWN")
            groups.setdefault(s, []).append(t)
    else:
        groups = {"ALL": list(window.columns)}

    pairs = []
    for group_name, tickers in groups.items():
        norm = {}
        for t in tickers:
            s = window[t].dropna()
            if len(s) < min_history:
                continue
            norm[t] = s / s.iloc[0]
        valid = list(norm.keys())
        if len(valid) < 2:
            continue

        candidates = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = valid[i], valid[j]
                na, nb = norm[a].align(norm[b], join="inner")
                if len(na) < min_history:
                    continue
                spread = na - nb
                ssd = float((spread ** 2).sum())
                candidates.append((a, b, ssd, float(spread.std())))
        candidates.sort(key=lambda x: x[2])
        limit = top_n_global if not restrict_to_sector else top_n_per_sector
        for a, b, ssd, spread_std in candidates[:limit]:
            if spread_std > 1e-6:
                pairs.append((a, b, spread_std))
    return pairs


def backtest_pairs(price_df: pd.DataFrame, sector_map: dict,
                    formation_days: int = 252, trading_days: int = 63,
                    top_n_per_sector: int = 2, entry_z: float = 2.0,
                    exit_z: float = 0.5, stop_z: float = 4.0,
                    cost_bi: float = 0.003,
                    restrict_to_sector: bool = True, top_n_global: int = 20):
    """按季度滚动形成期+交易期，返回逐个交易期的汇总(list of dict)，
    以及每期内每笔配对交易的明细(供诊断用)。"""
    index = price_df.index
    n = len(index)
    period_results = []
    all_trades = []

    period_start = formation_days
    while period_start + trading_days <= n:
        formation_end_idx = period_start
        trading_end_idx = period_start + trading_days

        pairs = select_pairs(price_df, sector_map, formation_end_idx,
                              formation_days=formation_days, top_n_per_sector=top_n_per_sector,
                              restrict_to_sector=restrict_to_sector, top_n_global=top_n_global)

        pair_rets = []
        for a, b, spread_std in pairs:
            base_a_hist = price_df[a].iloc[:formation_end_idx].dropna()
            base_b_hist = price_df[b].iloc[:formation_end_idx].dropna()
            if len(base_a_hist) == 0 or len(base_b_hist) == 0:
                continue
            base_a, base_b = base_a_hist.iloc[-1], base_b_hist.iloc[-1]

            trading_window = price_df.iloc[formation_end_idx:trading_end_idx]
            pa = trading_window[a].dropna()
            pb = trading_window[b].dropna()
            common = pa.index.intersection(pb.index)
            if len(common) < 5:
                continue
            pa, pb = pa.loc[common], pb.loc[common]
            na = pa / base_a
            nb = pb / base_b
            spread = na - nb
            z = spread / spread_std

            position = 0
            entry_na = entry_nb = None
            for i in range(len(z)):
                if position == 0:
                    if z.iloc[i] > entry_z:
                        position = -1
                        entry_na, entry_nb = na.iloc[i], nb.iloc[i]
                    elif z.iloc[i] < -entry_z:
                        position = 1
                        entry_na, entry_nb = na.iloc[i], nb.iloc[i]
                else:
                    ret_a = na.iloc[i] / entry_na - 1
                    ret_b = nb.iloc[i] / entry_nb - 1
                    pair_ret = position * (ret_a - ret_b) / 2
                    exit_now = (abs(z.iloc[i]) < exit_z) or (abs(z.iloc[i]) > stop_z) or (i == len(z) - 1)
                    if exit_now:
                        net_ret = pair_ret - cost_bi
                        pair_rets.append(net_ret)
                        all_trades.append({
                            "pair": f"{a}/{b}", "date": common[i].strftime("%Y-%m-%d"), "ret": net_ret,
                        })
                        position = 0
                        entry_na = entry_nb = None

        period_results.append({
            "period_start_date": index[formation_end_idx].strftime("%Y-%m-%d"),
            "period_end_date": index[min(trading_end_idx, n) - 1].strftime("%Y-%m-%d"),
            "n_pairs_selected": len(pairs),
            "n_trades": len(pair_rets),
            "period_ret": float(np.mean(pair_rets)) if pair_rets else 0.0,
        })
        period_start += trading_days

    return period_results, all_trades


def metrics_from_periods(period_results, periods_per_year: float = 4.0):
    rets = np.array([p["period_ret"] for p in period_results])
    if len(rets) < 2:
        return {"n_periods": len(rets), "CAGR": None, "Sharpe": None, "MaxDD": None}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_years = len(rets) / periods_per_year
    cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year))
    return {"n_periods": len(rets), "CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd,
            "avg_n_trades_per_period": float(np.mean([p["n_trades"] for p in period_results]))}
