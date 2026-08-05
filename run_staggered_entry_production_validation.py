"""
Cycle67大修大补①(续，正式版)：分批建仓三窗口验证——用真正的生产链路
(`run_walk_forward_backtest`，完整walk-forward重新训练，不是post-hoc
用缓存picks重新结算)，因为post-hoc重建版本(见`run_staggered_entry_
3window_validation.py`)算出来的基线Sharpe(2022-2026约1.6)跟这次会话
已确认的生存偏差修正后基线(约-0.05)差距过大，不能直接采信，必须用
能跟已知基线数字对得上的真实生产路径重新验证一遍，才能信任"分批建仓
更好"这个结论。

给`backtest.py`新增了`staggered_entry_days`参数(默认None，向后兼容，
已做函数级回归测试确认默认行为不变)，这次分别用`staggered_entry_days=
None`(基线)和`=3`(周一二三分批)跑完整三段区间。
"""
from __future__ import annotations

import pickle

from backtest import run_walk_forward_backtest
from optimized_metrics_v4 import run_backtest_with_metrics
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]


def main():
    for label, cache_path, is_pit in PERIODS:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]

        membership = None
        if is_pit:
            membership = compute_point_in_time_membership(
                c_df.index, list(c_df.columns), cache["added_dates"], cache["removal_dates"])

        print(f"\n{'='*70}\n {label} - 基线(一次性买入)\n{'='*70}")
        results_base = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=8, xgb_n_jobs=1,
            enable_earnings_blackout=False,
            point_in_time_membership=membership,
            staggered_entry_days=None,
        )
        print(f"\n[{label} 基线报告]")
        run_backtest_with_metrics(results_base)

        print(f"\n{'='*70}\n {label} - 分批买入(周一二三)\n{'='*70}")
        results_stagger = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=8, xgb_n_jobs=1,
            enable_earnings_blackout=False,
            point_in_time_membership=membership,
            staggered_entry_days=3,
        )
        print(f"\n[{label} 分批买入报告]")
        run_backtest_with_metrics(results_stagger)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
