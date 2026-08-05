"""
Alpha 分解：宏观熔断规则（check_macro_warning）到底贡献了多少超额收益？

跑两遍同样的历史数据 —— 一遍开着熔断（现有默认行为），一遍完全关掉熔断、
让 XGBoost 选股结果无论行情好坏都照常执行 —— 对比两者的 CAGR / Sharpe /
最大回撤。如果"关熔断"跟"开熔断"差距很小，说明超额收益主要来自选股模型
本身；如果差距很大，说明现在报告里的超额收益大部分是"行情不好就空仓"这条
简单规则贡献的，选股模型本身的价值需要打个问号。
"""
from __future__ import annotations

from backtest import fetch_market_data, run_walk_forward_backtest
from stat_validation import compute_period_metrics


def _run(c_df, o_df, h_df, l_df, v_df, vix, enable_macro_filter: bool):
    label = "开启熔断（默认行为）" if enable_macro_filter else "关闭熔断（纯选股模型）"
    print(f"\n{'='*70}\n  跑一遍：{label}\n{'='*70}")
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, enable_macro_filter=enable_macro_filter,
    )
    import pandas as pd
    df = pd.DataFrame(results)
    df["weekly_return"] = df["weekly_return"].fillna(0.0)
    df["net_worth"] = df["net_worth"].ffill().fillna(100.0)
    rf_weekly = (1 + 0.04) ** (1 / 52) - 1
    metrics = compute_period_metrics(df, rf_weekly, weeks_per_year=52)
    metrics["label"] = label
    return metrics


if __name__ == "__main__":
    c_df, o_df, h_df, l_df, v_df, vix, _ = fetch_market_data()

    with_filter = _run(c_df, o_df, h_df, l_df, v_df, vix, enable_macro_filter=True)
    without_filter = _run(c_df, o_df, h_df, l_df, v_df, vix, enable_macro_filter=False)

    print(f"\n{'='*70}")
    print(" 【熔断规则贡献分解】")
    print(f"{'='*70}")
    for m in (with_filter, without_filter):
        print(f" {m['label']:22s}  CAGR {m['cagr']*100:6.2f}%  "
              f"Sharpe {m['sharpe']:5.2f}  最大回撤 {m['max_dd']*100:6.2f}%  "
              f"胜率 {m['win_rate']*100:5.1f}%  ({m['n_weeks']}周)")
    cagr_gap = with_filter["cagr"] - without_filter["cagr"]
    sharpe_gap = with_filter["sharpe"] - without_filter["sharpe"]
    print(f"\n 熔断规则带来的增量: CAGR {cagr_gap*100:+.2f}pp, Sharpe {sharpe_gap:+.2f}")
    if abs(sharpe_gap) > without_filter["sharpe"] * 0.5 and without_filter["sharpe"] > 0:
        print(" -> 熔断规则贡献了 Sharpe 里相当大的一块，选股模型单独的价值需要打个问号。")
    else:
        print(" -> 两者差距不算悬殊，超额收益看起来更多是选股模型本身贡献的。")
    print("=" * 70)
