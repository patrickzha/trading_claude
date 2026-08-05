"""
按市值分层验证——现在500只票混着训练一个模型，大盘股和成长股的动量规律
可能不一样。市值数据不用现拉yfinance（.info调用慢、500只票容易被限流），
复用已经缓存好的SEC EDGAR基本面数据里的 shares_outstanding 字段，
市值 = 最新股价 × 最新股本数，静态近似（不是逐周点位正确的市值，可能有轻微
前视偏差，但用于"分层是否有效"这个方向性判断足够）。

不需要改动 us_stock_screener.py / backtest.py 任何代码——直接把价格
DataFrame切片到每层的股票子集，用已经验证过的默认配置（止损2%+RSI阈值100+
新近度加权，都是模块默认值）分别跑一遍，比较大盘股层/中盘层/小盘层各自的
表现，看是不是某个市值区间的信号明显更强。
"""
from __future__ import annotations

import json
import pickle
import os

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"
SEC_CACHE = "/Users/zhang/Desktop/trading/sec_fundamentals_cache.json"


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        full["c"], full["o"], full["h"], full["l"], full["v"], full["vix"], full.get("spy")
    )
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)
    with open(SEC_CACHE) as f:
        sec_cache = json.load(f)

    # 数据质量排查发现：部分票的 shares_outstanding 最新记录严重过期或明显错误
    # （比如BRK-B最新一条是2011年的、FOXA/FOX的值是"1股"这种打标签错误），
    # 直接拿"最后一条"当最新市值会算出离谱结果（BRK-B、IBKR掉进小盘层）。
    # 加两道过滤：披露日期超过2年不可信；股数小于等于1000这种明显错误值跳过。
    latest_price = c_df.iloc[-1]
    latest_date = c_df.index[-1]
    market_caps = {}
    stale_skipped = 0
    for ticker in c_df.columns:
        if ticker not in sec_cache or "shares_outstanding" not in sec_cache[ticker]:
            continue
        shares_entries = sec_cache[ticker]["shares_outstanding"]
        if not shares_entries:
            continue
        latest_entry = shares_entries[-1]
        try:
            filed_date = pd.Timestamp(latest_entry["filed"])
        except Exception:
            continue
        age_days = (latest_date - filed_date).days
        if age_days > 730:
            stale_skipped += 1
            continue
        latest_shares = latest_entry["val"]
        price = latest_price.get(ticker)
        if price is None or np.isnan(price) or latest_shares <= 1000:
            continue
        market_caps[ticker] = price * latest_shares
    print(f"因股本数据过期(>2年未更新)跳过: {stale_skipped}只")

    print(f"有市值数据的票数: {len(market_caps)}/{len(c_df.columns)}")
    sorted_tickers = sorted(market_caps.keys(), key=lambda t: market_caps[t], reverse=True)
    n = len(sorted_tickers)
    tier_size = n // 3
    tiers = {
        "大盘层(市值前1/3)": sorted_tickers[:tier_size],
        "中盘层(市值中1/3)": sorted_tickers[tier_size:2*tier_size],
        "小盘层(市值后1/3)": sorted_tickers[2*tier_size:],
    }
    for name, tickers in tiers.items():
        caps = [market_caps[t] for t in tickers]
        print(f"{name}: {len(tickers)}只票, 市值范围 ${min(caps)/1e9:.1f}B ~ ${max(caps)/1e9:.1f}B")

    n_workers = max(1, os.cpu_count() or 1)

    def bench_for(used_dates):
        if spy_close is None:
            return None
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c_df.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)
        return bench_rets

    for tier_name, tickers in tiers.items():
        print(f"\n{'='*20} {tier_name} ({len(tickers)}只票) {'='*20}")
        c2, o2, h2, l2, v2 = c_df[tickers], o_df[tickers], h_df[tickers], l_df[tickers], v_df[tickers]
        results = run_walk_forward_backtest(
            c2, o2, h2, l2, v2, vix,
            enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
            earnings_calendar=earnings_calendar,
        )
        used_dates = {r["date"] for r in results}
        bench_rets = bench_for(used_dates)
        print(f"\n[{tier_name} 报告]")
        run_backtest_with_metrics(results, benchmark_returns=bench_rets)


if __name__ == "__main__":
    main()
