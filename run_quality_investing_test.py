"""
质量因子策略首次测试：三段历史区间(2014-2020/2009-2014用PIT修正后
membership，2022-2026用已有的点位时间正确554只股票池)，跟已有的价值腿
对比表现，重点看相关系数——如果质量因子跟价值因子高度相关，说明只是
"换个壳子的价值因子"，不算真正独立的新维度；如果相关系数低，才是这次
框架跳跃真正有意义的地方。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from quality_investing import backtest_quality_strategy
from value_investing import backtest_value_strategy
from combo_strategy import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]

with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


for label, price_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

    q_results, q_nav = backtest_quality_strategy(price_df, membership, spy_close, rebalance_days=126, top_n=30)
    v_results, v_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=126, top_n=30)

    q_cagr, q_sharpe, q_mdd = metrics(q_nav)
    v_cagr, v_sharpe, v_mdd = metrics(v_nav)

    q_df = pd.DataFrame(q_nav, columns=["date", "nav"]).set_index("date")["nav"]
    v_df = pd.DataFrame(v_nav, columns=["date", "nav"]).set_index("date")["nav"]
    combined = pd.DataFrame({"quality": q_df, "value": v_df}).dropna()
    corr = combined.pct_change().dropna().corr().loc["quality", "value"]

    print(f"\n=== {label} ===")
    print(f"  质量因子: CAGR={q_cagr*100:.2f}%, Sharpe={q_sharpe:.3f}, MaxDD={q_mdd*100:.2f}%, 候选数(首期)={q_results[0]['n_picks'] if q_results else 'N/A'}")
    print(f"  价值因子(对照): CAGR={v_cagr*100:.2f}%, Sharpe={v_sharpe:.3f}, MaxDD={v_mdd*100:.2f}%")
    print(f"  质量因子跟价值因子相关系数: {corr:.3f}")
