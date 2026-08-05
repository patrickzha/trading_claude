"""
质量因子(quality investing)选股策略——这次会话第一次真正跳出"技术特征+
XGBoost回归"这个框架的尝试(用户2026-08明确要求："以后的大修大补至少要
包含一个框架改动"，已写入CLAUDE.md)。

学术依据：Novy-Marx(2013)等研究发现，毛利率高的公司即使估值不便宜，
未来收益依然显著跑赢市场，是独立于价值因子的一个维度；Sloan(1996)的
应计利润异象发现，经营现金流跟净利润差距大(利润"含金量"低，可能是激进
会计处理)的公司未来收益更差；资产增长异象(Cooper et al. 2008)发现，
资产扩张过快的公司(可能是过度投资)未来收益也更差。这次用三个质量维度
构建综合分：毛利率(越高越好)、经营利润率(越高越好)、应计利润比率
(净利润-经营现金流，除以总资产，越低/越负越好)。

数据来源：复用`fetch_sec_fundamentals.py`已有的net_income缓存 +
`fetch_sec_quality_fundamentals.py`新抓的营收/毛利/经营利润/经营现金流/
总资产(SEC EDGAR Company Facts API，时间点数据用filed日期回溯，没有
前视偏差，方法论跟value_investing.py完全一致)。

如实说明数据局限：gross_profit这个科目在抓取时覆盖率只有44.8%(不是
所有公司都用XBRL标准标签披露毛利润)，比net_income/total_assets这些
基础科目覆盖率低得多——用综合排名分时，缺失毛利率的票不会被直接排除，
但会在那一项排名上处于中位(用可用维度的均值填充排名分，不是强行拉到
最差)，如实标注这个处理方式，不假装数据完整。
"""
from __future__ import annotations

import bisect
import json
import os

import numpy as np
import pandas as pd

_QUALITY_CACHE = None
_NET_INCOME_CACHE = None


def _load_quality_fundamentals() -> dict:
    global _QUALITY_CACHE
    if _QUALITY_CACHE is not None:
        return _QUALITY_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_quality_fundamentals_cache.json")
    try:
        with open(json_path) as f:
            raw = json.load(f)
        processed = {}
        for ticker, fields in raw.items():
            processed[ticker] = {}
            for metric, entries in fields.items():
                sorted_entries = sorted(entries, key=lambda e: e["filed"])
                processed[ticker][metric] = {
                    "filed_dates": [e["filed"] for e in sorted_entries],
                    "values": [e["val"] for e in sorted_entries],
                }
        _QUALITY_CACHE = processed
    except Exception as e:
        print(f"  [警告] 未能加载 sec_quality_fundamentals_cache.json（{type(e).__name__}: {e}）")
        _QUALITY_CACHE = {}
    return _QUALITY_CACHE


def _load_net_income() -> dict:
    """复用value_investing.py同一份net_income缓存(fetch_sec_fundamentals.py产出)。"""
    global _NET_INCOME_CACHE
    if _NET_INCOME_CACHE is not None:
        return _NET_INCOME_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_fundamentals_cache.json")
    try:
        with open(json_path) as f:
            raw = json.load(f)
        processed = {}
        for ticker, fields in raw.items():
            if "net_income" not in fields:
                continue
            sorted_entries = sorted(fields["net_income"], key=lambda e: e["filed"])
            processed[ticker] = {
                "filed_dates": [e["filed"] for e in sorted_entries],
                "values": [e["val"] for e in sorted_entries],
            }
        _NET_INCOME_CACHE = processed
    except Exception:
        _NET_INCOME_CACHE = {}
    return _NET_INCOME_CACHE


def _lookup_latest(field_data: dict, decision_date_str: str) -> float | None:
    filed_dates = field_data["filed_dates"]
    idx = bisect.bisect_right(filed_dates, decision_date_str) - 1
    if idx < 0:
        return None
    return field_data["values"][idx]


def _lookup_asof(field_data: dict, target_date_str: str) -> float | None:
    """找 filed <= target_date_str 里最新的一条——跟_lookup_latest逻辑一样，
    但用于查"一年前"这类历史目标日期，命名区分开只是为了调用处更清楚意图。"""
    return _lookup_latest(field_data, target_date_str)


def _compute_quality_features(ticker: str, decision_date) -> dict:
    """毛利率、经营利润率、应计利润比率(越低越好)、资产增长率(越低越好)，
    全部用SEC披露日期做时间点回溯。缺数据时对应特征给np.nan。"""
    qcache = _load_quality_fundamentals()
    nicache = _load_net_income()
    nan = float("nan")
    result = {"gross_margin": nan, "operating_margin": nan, "accrual_ratio": nan, "asset_growth": nan}

    q = qcache.get(ticker)
    ni = nicache.get(ticker)
    decision_str = decision_date.strftime("%Y-%m-%d")
    one_year_ago_str = (decision_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    revenues = _lookup_latest(q["revenues"], decision_str) if q and "revenues" in q else None
    gross_profit = _lookup_latest(q["gross_profit"], decision_str) if q and "gross_profit" in q else None
    operating_income = _lookup_latest(q["operating_income"], decision_str) if q and "operating_income" in q else None
    op_cash_flow = _lookup_latest(q["operating_cash_flow"], decision_str) if q and "operating_cash_flow" in q else None
    total_assets = _lookup_latest(q["total_assets"], decision_str) if q and "total_assets" in q else None
    total_assets_prior = _lookup_asof(q["total_assets"], one_year_ago_str) if q and "total_assets" in q else None
    net_income = _lookup_latest(ni, decision_str) if ni else None

    if revenues is not None and revenues > 0:
        if gross_profit is not None:
            result["gross_margin"] = gross_profit / revenues
        if operating_income is not None:
            result["operating_margin"] = operating_income / revenues

    if net_income is not None and op_cash_flow is not None and total_assets is not None and total_assets > 0:
        result["accrual_ratio"] = (net_income - op_cash_flow) / total_assets

    if total_assets is not None and total_assets_prior is not None and total_assets_prior > 0:
        result["asset_growth"] = (total_assets - total_assets_prior) / total_assets_prior

    return result


def select_quality_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                              top_n: int = 30, min_fields: int = 2):
    """按综合质量分选股。min_fields：四个质量维度里至少要有几个非缺失才
    参与排名，避免只有1个维度有数据的票靠运气排到极端位置。"""
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:rebalance_date].dropna()
        if len(px_series) == 0:
            continue
        feat = _compute_quality_features(t, rebalance_date)
        n_valid = sum(1 for v in feat.values() if not np.isnan(v))
        if n_valid < min_fields:
            continue
        row = {"ticker": t}
        row.update(feat)
        rows.append(row)

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows)
    # 排名：毛利率/经营利润率越高越好，应计比率/资产增长率越低越好。
    # 用pct=True算百分位排名(0~1)，缺失值排名后用当列均值(≈0.5)填充，
    # 不因为某一维度缺失就把票直接判死刑，也不让缺失变相占到"最优"的便宜。
    for col, ascending in [("gross_margin", False), ("operating_margin", False),
                            ("accrual_ratio", True), ("asset_growth", True)]:
        rank_col = f"rank_{col}"
        df[rank_col] = df[col].rank(ascending=ascending, pct=True)
        df[rank_col] = df[rank_col].fillna(df[rank_col].mean())

    df["composite"] = df[[f"rank_{c}" for c in ["gross_margin", "operating_margin", "accrual_ratio", "asset_growth"]]].mean(axis=1)
    df = df.sort_values("composite", ascending=False)  # 排名分越高(越接近1)越好
    return df["ticker"].head(top_n).tolist()


def backtest_quality_strategy(price_df: pd.DataFrame, membership: dict, spy_close: pd.Series | None,
                               rebalance_days: int = 126, top_n: int = 30, cost_bi: float = 0.003):
    """半年度调仓的质量因子组合回测，方法论跟backtest_value_strategy完全
    一致(逐日重建净值、turnover-based成本)，方便直接横向对比。"""
    index = price_df.index
    n = len(index)
    results = []
    daily_nav = []

    rebalance_idx = 252
    running_nav = 1.0
    prev_picks: set[str] = set()
    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        next_date = index[rebalance_idx + rebalance_days]

        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_quality_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n)

        picks_set = set(picks)
        turnover = len(picks_set - prev_picks) / len(picks_set) if picks_set else 0.0
        running_nav *= (1 - turnover * cost_bi)
        prev_picks = picks_set

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        for d in period_days:
            day_navs = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_navs.append(p / p0)
            if day_navs:
                daily_nav.append((d, running_nav * float(np.mean(day_navs))))
        if daily_nav:
            running_nav = daily_nav[-1][1]

        rets = []
        for t in picks:
            try:
                p0 = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
                p1 = price_df[t].loc[:next_date].dropna().iloc[-1]
                rets.append(p1 / p0 - 1)
            except (IndexError, KeyError):
                continue

        spy_ret = None
        if spy_close is not None:
            try:
                s0 = spy_close.loc[:rebalance_date].dropna().iloc[-1]
                s1 = spy_close.loc[:next_date].dropna().iloc[-1]
                spy_ret = float(s1 / s0 - 1)
            except (IndexError, KeyError):
                pass

        results.append({
            "rebalance_date": rebalance_date.strftime("%Y-%m-%d"),
            "n_picks": len(picks),
            "period_ret": float(np.mean(rets)) if rets else 0.0,
            "spy_ret": spy_ret,
        })
        rebalance_idx += rebalance_days

    return results, daily_nav
