"""
Cycle34大修大补②(框架改动，新组合分配机制)：动态因子腿配置——用trailing
6个月已实现Sharpe动态调整动量/价值/低波动三条腿在组合里的资金权重，
代替现有combo_strategy.py固定1/3等权混合。这是"策略层面的资金再分配"，
跟第170条(个股层面直接构造复合分)、min_variance/risk_parity/hrp(给定
选股结果后的个股仓位分配)都是不同维度的组合构建改动——这次是给三个
已经各自选股完成的子策略动态分配资金，理论依据是"表现较好的因子在
未来一段时间可能有动量延续"(因子轮动的学术依据没有个股动量那么强，
如实标注这是相对较弱的先验)。

方法：每21个交易日(约1个月)重新平衡一次三条腿的资金权重，权重正比于
各腿trailing 126天(约6个月)已实现Sharpe(只用截止当前的历史数据，不用
未来信息)，Sharpe为负的腿权重floor到一个很小的正数(不完全清零，避免
单一负Sharpe期把该腿完全踢出可能错过后续反弹)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
LOOKBACK = 126
REBALANCE_EVERY = 21
MIN_WEIGHT_FLOOR = 0.05


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def sharpe_of(rets):
    if len(rets) < 5:
        return 0.0
    return float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))


def dynamic_weight_combo(leg_nav: pd.DataFrame) -> pd.Series:
    leg_rets = leg_nav.pct_change().dropna()
    n = len(leg_rets)
    weights_over_time = pd.DataFrame(index=leg_rets.index, columns=leg_rets.columns, dtype=float)
    current_weights = pd.Series(1.0 / leg_rets.shape[1], index=leg_rets.columns)

    for i in range(n):
        if i > 0 and i % REBALANCE_EVERY == 0 and i >= LOOKBACK:
            window = leg_rets.iloc[i - LOOKBACK:i]
            sharpes = window.apply(sharpe_of)
            sharpes_clipped = sharpes.clip(lower=MIN_WEIGHT_FLOOR)
            current_weights = sharpes_clipped / sharpes_clipped.sum()
        weights_over_time.iloc[i] = current_weights

    portfolio_rets = (leg_rets * weights_over_time).sum(axis=1)
    portfolio_nav = (1 + portfolio_rets).cumprod()
    return portfolio_nav


def metrics_from_nav(nav: pd.Series):
    rets = nav.pct_change().dropna()
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = sharpe_of(rets)
    mdd = true_mdd(nav.values)
    return cagr, sharpe, mdd


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
         f"{SCRATCH}/default_full_results_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
         f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
         f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
    ]

    for label, price_cache_path, mom_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label} ===")
        report = run_combo_strategy(price_df, membership, spy_close, mom_results)
        leg_nav = report["leg_nav"]

        equal_cagr, equal_sharpe, equal_mdd = report["CAGR"], report["Sharpe"], report["MaxDD"]
        dyn_nav = dynamic_weight_combo(leg_nav)
        dyn_cagr, dyn_sharpe, dyn_mdd = metrics_from_nav(dyn_nav)

        print(f"  基线(固定等权1/3/1/3/1/3): CAGR={equal_cagr*100:.2f}%, Sharpe={equal_sharpe:.3f}, MaxDD={equal_mdd*100:.2f}%")
        print(f"  动态trailing-Sharpe加权: CAGR={dyn_cagr*100:.2f}%, Sharpe={dyn_sharpe:.3f}, MaxDD={dyn_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
