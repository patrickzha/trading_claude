"""
阶段E诊断类分析：不是候选验证，是给当前已采纳的默认配置（止损2% + RSI阈值100
+ 新近度加权26周）做几个诊断，帮忙判断这套系统的风险画像和选股逻辑靠不靠谱：

1. 行业/风格集中度检查：回测期间实际选中的候选是不是系统性偏向某个行业——
   如果模型学到的其实是"买科技股"这么简单粗暴的规律，不是真的动量信号，
   这里应该能看出来。
2. 蒙特卡洛重采样风险画像：历史回测只是一条已经发生的路径，用block bootstrap
   重采样生成大量可能的"平行世界"收益路径，估计"连续亏损超过X%的概率"这类
   历史最大回撤这一个数字看不出来的风险。
3. 只用2022年（标普500全年下跌约-18%的熊市年份）单独跑一遍——已经采纳的
   三个改进是不是只在过去5年这个整体上涨的环境里管用，熊市年份表现如何。
"""
from __future__ import annotations

import pickle
import os

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from us_stock_screener import SECTOR_MAP
from optimized_metrics_v4 import run_backtest_with_metrics

DATA_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/sp500_data_cache.pkl"
EARNINGS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/earnings_calendar_cache.pkl"
RESULTS_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/default_full_results_cache.pkl"


def sector_concentration_check(results):
    print("\n" + "=" * 70)
    print(" 诊断1：实际选中候选的行业分布")
    print("=" * 70)
    sector_counts = {}
    total_picks = 0
    for r in results:
        for ticker in r["picks"]:
            sec = SECTOR_MAP.get(ticker, "UNKNOWN")
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            total_picks += 1
    if total_picks == 0:
        print("  没有任何候选，跳过。")
        return
    for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec:10s}: {cnt:4d}次 ({cnt/total_picks:.1%})")
    max_sector_share = max(sector_counts.values()) / total_picks
    print(f"\n  最大单一行业占比: {max_sector_share:.1%}"
          f"（标普500里科技+可选消费合计占比常年在40%左右做参考基准）")


def monte_carlo_risk_profile(results, n_sims=5000, block_size=4, seed=42):
    print("\n" + "=" * 70)
    print(" 诊断2：Block Bootstrap 蒙特卡洛风险画像")
    print("=" * 70)
    rng = np.random.default_rng(seed)
    weekly_rets = np.array([r["weekly_return"] for r in results])
    n = len(weekly_rets)
    n_blocks_needed = int(np.ceil(n / block_size))

    worst_drawdowns = []
    final_navs = []
    for _ in range(n_sims):
        idx = []
        for _ in range(n_blocks_needed):
            start = rng.integers(0, n - block_size + 1)
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        path = weekly_rets[idx]
        nav = np.cumprod(1 + path)
        peak = np.maximum.accumulate(nav)
        dd = (nav - peak) / peak
        worst_drawdowns.append(dd.min())
        final_navs.append(nav[-1] - 1)

    worst_drawdowns = np.array(worst_drawdowns)
    final_navs = np.array(final_navs)
    print(f"  模拟次数: {n_sims}, 每次路径长度: {n}周（跟实际回测周数一致）")
    print(f"  最大回撤分布: 中位数 {np.median(worst_drawdowns):.1%}, "
          f"5%分位(更差) {np.percentile(worst_drawdowns, 5):.1%}, "
          f"1%分位(极端更差) {np.percentile(worst_drawdowns, 1):.1%}")
    for threshold in [-0.20, -0.25, -0.30, -0.35]:
        prob = (worst_drawdowns <= threshold).mean()
        print(f"  P(最大回撤 <= {threshold:.0%}): {prob:.1%}")
    print(f"  累计收益分布: 中位数 {np.median(final_navs):.1%}, "
          f"5%分位(更差) {np.percentile(final_navs, 5):.1%}")
    print(f"  P(累计收益 <= 0，即整体亏钱): {(final_navs <= 0).mean():.1%}")


def bear_market_2022_check(spy_close_full_index):
    print("\n" + "=" * 70)
    print(" 诊断3：仅2022年（标普500全年约-18%的熊市）单独验证")
    print("=" * 70)
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = full["c"], full["o"], full["h"], full["l"], full["v"], full["vix"]
    spy_close = full.get("spy")

    # 2022年前面留够120天lookback缓冲
    mask = (c_df.index >= "2021-09-01") & (c_df.index <= "2022-12-31")
    c2, o2, h2, l2, v2, vix2 = c_df.loc[mask], o_df.loc[mask], h_df.loc[mask], l_df.loc[mask], v_df.loc[mask], vix.loc[mask]

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c2, o2, h2, l2, v2, vix2,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
    )
    bench_rets = None
    if spy_close is not None:
        used_dates = {r["date"] for r in results}
        bench_rets = []
        for monday, friday in get_weekly_trading_dates(c2.index):
            if monday.strftime("%Y-%m-%d") not in used_dates:
                continue
            try:
                bench_rets.append((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday])
            except Exception:
                bench_rets.append(0.0)
    print("\n[2022熊市年份单独验证报告]")
    run_backtest_with_metrics(results, benchmark_returns=bench_rets)


def main():
    with open(DATA_CACHE, "rb") as f:
        full = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        full["c"], full["o"], full["h"], full["l"], full["v"], full["vix"], full.get("spy")
    )
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    n_workers = max(1, os.cpu_count() or 1)
    print("跑一次当前默认配置的全量5年回测，拿到原始 picks 数据供诊断用...")
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar,
    )
    with open(RESULTS_CACHE, "wb") as f:
        pickle.dump(results, f)
    print(f"已缓存原始结果到 {RESULTS_CACHE}")

    sector_concentration_check(results)
    monte_carlo_risk_profile(results)
    bear_market_2022_check(spy_close)


if __name__ == "__main__":
    main()
