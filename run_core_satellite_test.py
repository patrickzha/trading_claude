"""
核心+卫星双层账户结构：核心仓位(SPY买入持有) + 卫星仓位(现有周频动量策略)
按固定比例混合，而不是100%全押策略或100%空仓。这个想法不需要重新训练
模型——策略每周的收益率序列已经算出来了，混合是纯粹的组合数学：

    blended_ret[t] = core_weight * spy_ret[t] + (1 - core_weight) * strategy_ret[t]

用"验证结论"第33条同一份确定性代码跑一次baseline(点位时间正确的554只
股票池、当前默认配置)，拿到逐周收益率，跟同一段时间的SPY逐周收益率按
几个比例混合，看Sharpe/回撤有没有改善。

如实说：这个测试本质上是在测"用被动指数敞口稀释一个edge接近0的策略，
能不能机械性地改善风险调整后收益"，不是在测"策略本身变强了"——如果
策略的Sharpe本来就接近0、跟SPY相关性不是特别高，按组合数学混入正
Sharpe的SPY几乎必然会让整体Sharpe上升，这本身不构成"选股模型有效"的
证据，报告里会如实标注这个局限。
"""
from __future__ import annotations

import pickle

import numpy as np

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH, EARNINGS_CACHE

CORE_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]  # 0.0=纯策略, 1.0=纯SPY


def metrics(rets):
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    n_weeks = len(rets)
    cagr = float(nav[-1] ** (52 / n_weeks) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": mdd}


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix, spy_close = (
        cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"], cache["spy"]
    )
    sector_map = cache["sector_map"]
    added_dates = cache["added_dates"]
    removal_dates_used = cache["removal_dates"]
    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, removal_dates_used)

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar_full = pickle.load(f)

    import os
    n_workers = max(1, os.cpu_count() or 1)
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar_full,
        sector_map=sector_map,
        point_in_time_membership=membership,
    )
    strategy_rets = [r["weekly_return"] for r in results]

    used_dates = {r["date"] for r in results}
    spy_rets = []
    for monday, friday in get_weekly_trading_dates(c_df.index):
        if monday.strftime("%Y-%m-%d") not in used_dates:
            continue
        try:
            spy_rets.append(float((spy_close.loc[friday] - spy_close.loc[monday]) / spy_close.loc[monday]))
        except Exception:
            spy_rets.append(0.0)

    assert len(strategy_rets) == len(spy_rets), f"{len(strategy_rets)} vs {len(spy_rets)}"

    print(f"\n共 {len(strategy_rets)} 周")
    print(f"{'core_weight':>12} | {'CAGR':>8} | {'Sharpe':>8} | {'MaxDD':>8}")
    for cw in CORE_WEIGHTS:
        blended = [cw * s + (1 - cw) * strat for s, strat in zip(spy_rets, strategy_rets)]
        m = metrics(blended)
        label = f"{cw:.1f}(纯SPY)" if cw == 1.0 else (f"{cw:.1f}(纯策略)" if cw == 0.0 else f"{cw:.1f}")
        print(f"{label:>12} | {m['CAGR']*100:7.2f}% | {m['Sharpe']:8.3f} | {m['MaxDD']*100:7.2f}%")


if __name__ == "__main__":
    main()
