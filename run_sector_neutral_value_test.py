"""
Cycle68大修大补②(框架改动，新组合构造方式——行业中性排名)：价值因子的
P/E、P/B、ROE排名从"全市场排名"改成"行业内部排名"，去掉价值因子对金融/
能源/工业这类传统低估值行业的结构性偏好，只保留"在自己所在行业里相对
便宜/优质"这个信息。用第276条已验证的最终配置(HRP权重+3%止损+周频
调仓)作为底座，三段区间对比。
"""
from __future__ import annotations

import pickle

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

BASE = dict(value_weighting="hrp", value_rebalance_days=5,
            value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

# select_value_portfolio目前没有sector_neutral透传通道(backtest_value_
# strategy/run_combo_strategy/run_full_system都没有开放这个参数)，这次
# 用monkeypatch方式在select_value_portfolio层面直接测试独立效果，不
# 改动生产链路(如果结果有希望，再决定要不要正式打通)。
import value_investing
from us_stock_screener import SECTOR_MAP

_original_select = value_investing.select_value_portfolio


def _sector_neutral_select(price_df, valid_tickers, rebalance_date, top_n=30, **kwargs):
    kwargs.pop("sector_map", None)
    return _original_select(price_df, valid_tickers, rebalance_date, top_n=top_n,
                             sector_neutral=True, sector_map=SECTOR_MAP_HOLDER["map"], **kwargs)


SECTOR_MAP_HOLDER = {"map": None}

for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    # 历史区间缓存(2014-2020/2009-2014)没有存sector_map字段(如实标注这个
    # 数据缺口)，回退用当前的模块级SECTOR_MAP——这是这次会话反复用过的
    # 处理方式(比如第113条)，行业分类相对稳定，用"现在"的分类去标注历史
    # 区间的股票，误差应该有限，不是精确解但优于完全跳过测试。
    SECTOR_MAP_HOLDER["map"] = cache.get("sector_map") or SECTOR_MAP
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n{'='*70}\n {label}\n{'='*70}")

    value_investing.select_value_portfolio = _original_select
    r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
    print(f"  全市场排名(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%, CAGR={r_base['CAGR']*100:.2f}%")

    if SECTOR_MAP_HOLDER["map"] is None:
        print("  [警告] 无sector_map，跳过行业中性测试")
        continue

    value_investing.select_value_portfolio = _sector_neutral_select
    r_neutral = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
    print(f"  行业中性排名: Sharpe={r_neutral['Sharpe']:.3f}, MaxDD={r_neutral['MaxDD']*100:.2f}%, CAGR={r_neutral['CAGR']*100:.2f}%")
    value_investing.select_value_portfolio = _original_select

print("\n全部完成。")
