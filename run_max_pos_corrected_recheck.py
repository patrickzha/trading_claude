"""
Cycle26大修大补①：第7条("持仓数量放宽到8-12只，Sharpe两个窗口都变差，
不建议改")是这次会话很早期的结论，测试时用的是还没做纳入日期过滤的
候选池——现在核心候选质量结构已经因为PIT修正变了(phantom票被移除)，
第4-12名候选的"平均质量比前3名差多少"这个相对关系有没有可能也跟着变，
值得用PIT修正后的数据重新测一次，确认第7条的结论是否依然成立。
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from backtest import run_walk_forward_backtest
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")
N_WORKERS = max(1, os.cpu_count() or 1)

CONFIGS = {
    "max_pos=3(默认)": {"max_pos": 3},
    "max_pos=6": {"max_pos": 6},
    "max_pos=10": {"max_pos": 10},
}


def rough_metrics(results):
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    t0 = time.time()
    with open(f"{SCRATCH}/historical_2014_2020_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]

    import json
    import pandas as pd
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}
    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, {})

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    for cname, kwargs in CONFIGS.items():
        print(f"\n--- {cname} ---")
        results = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1, earnings_calendar=earnings_calendar,
            point_in_time_membership=membership,
            **kwargs,
        )
        m = rough_metrics(results)
        print(f"  {cname}: {m}")

    print(f"\n总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    main()
