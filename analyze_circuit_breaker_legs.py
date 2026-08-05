"""
熔断规则两条腿拆开验证：VIX突增 vs 大盘动量破位，到底是哪条在起作用？
====================================================================
上一轮 alpha 分解（analyze_macro_filter_contribution.py）证明了"熔断规则整体"
贡献了几乎全部历史正收益，但那次是把 VIX 腿和动量腿绑在一起测的。同时
Granger 检验（diagnostics_granger_vix.py）显示 VIX 变化本身对这个股票池的
未来收益没有显著领先关系。这里把两条腿拆开单独测，四种组合里"两腿都开"
和"两腿都关"已经在上一轮跑过（5年数据，236周）：
    两腿都开（当前默认）: CAGR  6.25%  Sharpe  0.20  最大回撤 -24.90%
    两腿都关（纯选股）  : CAGR -8.18%  Sharpe -0.35  最大回撤 -44.91%
这里只需要再跑"只留动量腿"和"只留VIX腿"这两个新组合，四个放一起对比，
就能确认到底是哪条腿在真正起作用。
"""
from __future__ import annotations

import time
import pandas as pd

from backtest import fetch_market_data, run_walk_forward_backtest
from stat_validation import compute_period_metrics

# 上一轮 run_full_validation.py 已经跑过的两个组合，直接复用结果，不重复计算
KNOWN_RESULTS = {
    "两腿都开（当前默认）": {"cagr": 0.0625, "sharpe": 0.20, "max_dd": -0.2490, "win_rate": 0.674, "n_weeks": 236},
    "两腿都关（纯选股）":   {"cagr": -0.0818, "sharpe": -0.35, "max_dd": -0.4491, "win_rate": 0.381, "n_weeks": 236},
}


def _run(c_df, o_df, h_df, l_df, v_df, vix, label, enable_vix_leg, enable_momentum_leg):
    print(f"\n{'='*70}\n  跑一遍：{label}\n{'='*70}")
    rf_weekly = (1 + 0.04) ** (1 / 52) - 1
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False,
        enable_vix_leg=enable_vix_leg, enable_momentum_leg=enable_momentum_leg,
    )
    df = pd.DataFrame(results)
    df["weekly_return"] = df["weekly_return"].fillna(0.0)
    df["net_worth"] = df["net_worth"].ffill().fillna(100.0)
    m = compute_period_metrics(df, rf_weekly, 52)
    m["label"] = label
    return m


if __name__ == "__main__":
    t0 = time.time()
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data(days=1825)

    momentum_only = _run(c_df, o_df, h_df, l_df, v_df, vix,
                         "只留动量腿", enable_vix_leg=False, enable_momentum_leg=True)
    vix_only = _run(c_df, o_df, h_df, l_df, v_df, vix,
                    "只留VIX腿", enable_vix_leg=True, enable_momentum_leg=False)

    print(f"\n{'='*70}")
    print(" 【熔断规则两条腿分解 — 四种组合对比】")
    print(f"{'='*70}")
    all_results = list(KNOWN_RESULTS.items()) + [
        ("只留动量腿", momentum_only), ("只留VIX腿", vix_only),
    ]
    for label, m in all_results:
        print(f" {label:20s}  CAGR {m['cagr']*100:6.2f}%  Sharpe {m['sharpe']:5.2f}  "
              f"最大回撤 {m['max_dd']*100:6.2f}%  胜率 {m['win_rate']*100:5.1f}%  ({m['n_weeks']}周)")

    print(f"\n耗时: {time.time()-t0:.1f}s")
