"""
市值分层那次全量测试发现一个干净的单调信号：标普500内部，市值越小的1/3，
这套策略表现越好（大盘层甚至跑输SPY），跟"换成标普400"失败的结论不矛盾——
不是"越小盘越好"，是标普500内部本身就有"大盘股定价太有效"这个规律。

那次是单次5年全量测试，这次用同样的三窗口方法论重新验证一遍"剔除市值最大
的1/3"这个具体候选，跟之前所有候选一样的标准，不能因为信号好看就跳过初筛
直接采纳。用已经修好的市值计算逻辑（SEC shares_outstanding，过滤掉2年以上
未更新的过期数据）。不需要改动 us_stock_screener.py / backtest.py——直接
在切片阶段排除掉大盘层的票。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from run_batch_candidates import slice_window, run_one, rough_metrics, DATA_CACHE, EARNINGS_CACHE

SEC_CACHE = "/Users/zhang/Desktop/trading/sec_fundamentals_cache.json"


def compute_valid_tickers_excl_top_tier(c_df: pd.DataFrame) -> list[str]:
    with open(SEC_CACHE) as f:
        sec_cache = json.load(f)
    latest_price = c_df.iloc[-1]
    latest_date = c_df.index[-1]
    market_caps = {}
    for ticker in c_df.columns:
        if ticker not in sec_cache or "shares_outstanding" not in sec_cache[ticker]:
            continue
        entries = sec_cache[ticker]["shares_outstanding"]
        if not entries:
            continue
        e = entries[-1]
        try:
            filed = pd.Timestamp(e["filed"])
        except Exception:
            continue
        if (latest_date - filed).days > 730:
            continue
        shares = e["val"]
        price = latest_price.get(ticker)
        if price is None or np.isnan(price) or shares <= 1000:
            continue
        market_caps[ticker] = price * shares

    sorted_t = sorted(market_caps.keys(), key=lambda t: market_caps[t], reverse=True)
    n = len(sorted_t)
    tier_size = n // 3
    excluded_top_tier = set(sorted_t[:tier_size])
    # 没有市值数据的票（约501-459=42只）默认保留，不因为数据缺失就误伤
    return [t for t in c_df.columns if t not in excluded_top_tier]


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    data = {k: full[k] for k in ["c", "o", "h", "l", "v", "vix"]}
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    valid_tickers = compute_valid_tickers_excl_top_tier(data["c"])
    print(f"剔除市值前1/3后剩余票数: {len(valid_tickers)}/{len(data['c'].columns)}")

    n_total = len(data["c"].index)
    window_ends = {"近25周": n_total, "中段窗口(约1年前)": n_total - 300, "早段窗口(约2年前)": n_total - 600}

    deltas = []
    for win_name, end_idx in window_ends.items():
        win = slice_window(data, end_idx)
        print(f"\n{'='*20} {win_name} {'='*20}")
        base_res = run_one(win, earnings_calendar, {}, 8)
        base_m = rough_metrics(base_res)
        print(f"  baseline(全部500只): {base_m}")

        win_filtered = {k: (v[valid_tickers] if k in ("c", "o", "h", "l", "v") else v) for k, v in win.items()}
        excl_res = run_one(win_filtered, earnings_calendar, {}, 8)
        excl_m = rough_metrics(excl_res)
        print(f"  剔除大盘层1/3: {excl_m}")
        delta = excl_m["sharpe"] - base_m["sharpe"]
        deltas.append(delta)

    print(f"\n三窗口Sharpe差值: {[round(d,2) for d in deltas]}")
    if all(d > 0 for d in deltas):
        print("-> 方向一致更好，晋级全量验证")
    elif all(d < 0 for d in deltas):
        print("-> 方向一致更差，不晋级")
    else:
        print("-> 方向不一致，不晋级")


if __name__ == "__main__":
    main()
