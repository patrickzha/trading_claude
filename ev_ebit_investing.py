"""
Cycle62大修大补②(框架改动，新数据维度——资本结构中性的价值构造)：企业价值
倍数(EV/EBIT)因子——现有价值因子(第38条)用P/E、P/B、ROE，这几个比率都是
"股权"层面的估值(市值/股权账面价值/股权收益)，天生受资本结构(债务多少)
影响：两家经营质量完全一样的公司，一家借了很多债、一家没有，P/E可能
看起来差很多，但这只是杠杆效应，不是真的更便宜或更贵。EV/EBIT(格林布拉特
"神奇公式"用的核心估值指标)把债务加回市值算出企业价值(EV)，用税前息前
利润(EBIT)算倍数，理论上是资本结构中性的估值方式，衡量的是"买下整个
公司(股权+债权)要花多少钱，相对它税前经营利润的倍数"，不受杠杆水平
干扰。

数据来源：EBIT用`operating_income`(经营利润)代理(复用`sec_quality_
fundamentals_cache.json`，第134条质量因子已经抓过)；EV = 市值 +
`liabilities`(总负债，复用`sec_fundamentals_cache.json`) - 没有单独的
"现金及等价物"字段做净额处理(如实标注这个简化：真正的EV应该减掉现金，
这次的EV会系统性偏高，尤其是现金储备多的公司，比如科技股，这次没有
修正这个偏差)。
"""
from __future__ import annotations

import bisect
import json
import os

import numpy as np
import pandas as pd

_SEC_CACHE = None
_QUALITY_CACHE = None


def _load_sec_cache() -> dict:
    global _SEC_CACHE
    if _SEC_CACHE is not None:
        return _SEC_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_fundamentals_cache.json")
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
    _SEC_CACHE = processed
    return _SEC_CACHE


def _load_quality_cache() -> dict:
    global _QUALITY_CACHE
    if _QUALITY_CACHE is not None:
        return _QUALITY_CACHE
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sec_quality_fundamentals_cache.json")
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
    return _QUALITY_CACHE


def _lookup_latest(field_data: dict | None, decision_date_str: str) -> float | None:
    if not field_data:
        return None
    filed_dates = field_data["filed_dates"]
    idx = bisect.bisect_right(filed_dates, decision_date_str) - 1
    if idx < 0:
        return None
    return field_data["values"][idx]


def compute_ev_ebit(ticker: str, decision_date, curr_price: float) -> float | None:
    sec = _load_sec_cache().get(ticker, {})
    qual = _load_quality_cache().get(ticker, {})
    decision_str = decision_date.strftime("%Y-%m-%d")

    shares = _lookup_latest(sec.get("shares_outstanding"), decision_str)
    liabilities = _lookup_latest(sec.get("liabilities"), decision_str)
    operating_income = _lookup_latest(qual.get("operating_income"), decision_str)

    if not (shares and shares > 0 and liabilities is not None and operating_income and operating_income > 0):
        return None
    market_cap = curr_price * shares
    ev = market_cap + liabilities
    if ev <= 0:
        return None
    return ev / operating_income


def select_ev_ebit_portfolio(price_df: pd.DataFrame, valid_tickers: list, rebalance_date,
                              top_n: int = 30, max_multiple: float = 60.0) -> list:
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px_series = price_df[t].loc[:rebalance_date].dropna()
        if len(px_series) == 0:
            continue
        curr_price = px_series.iloc[-1]
        multiple = compute_ev_ebit(t, rebalance_date, curr_price)
        if multiple is not None and 0 < multiple < max_multiple:
            rows.append({"ticker": t, "ev_ebit": multiple})

    if len(rows) < top_n:
        return [r["ticker"] for r in rows]
    df = pd.DataFrame(rows).sort_values("ev_ebit")
    return df["ticker"].head(top_n).tolist()
