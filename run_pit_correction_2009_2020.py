"""
Cycle21大修大补①：把已经用在2022-2026(point_in_time_universe_cache.pkl)
的生存偏差修正——纳入日期过滤(`sp500_membership_added_dates.json`，503只
票的纳入日期，最早到1957年，之前抓Wikipedia成分股表时已经拿到，不需要
重新抓)——第一次应用到2014-2020和2009-2014这两段独立历史区间。这两段
一直有一个已知、反复标注过的局限："用当前(2026年)的股票池名单去拉这段
更早的历史"，意味着78只后来才被纳入标普500的票，在没被纳入之前就已经
被当成合法候选，这是"验证结论"第13条记录过的同一类生存偏差问题，一直
没有修复。

只做"纳入日期过滤"这一半(过滤掉"这周决策时还没被纳入"的票)，不做"剔除票
回补"这一半(把当时在标普500、现在已经不在的公司重新加回候选池)——后者
需要抓取大量2009-2020年间被剔除的公司的历史价格数据，数据可得性本身
不确定(部分公司已经退市/被收购，yfinance可能没有数据)，工程量和risk都
远大于前一半，如实标注为超出这次的范围，不假装已经完整解决生存偏差问题。

注意：跑这个脚本期间不要改us_stock_screener.py/backtest.py。
"""
from __future__ import annotations

import json
import os
import pickle
import time

import pandas as pd

from backtest import run_walk_forward_backtest
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"
EARNINGS_CACHE = os.path.join(SCRATCH, "earnings_calendar_cache.pkl")
N_WORKERS = max(1, os.cpu_count() or 1)

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]


def rough_metrics(results):
    import numpy as np
    rets = np.array([r["weekly_return"] for r in results]) if results else np.array([])
    if len(rets) < 2:
        return {"cum": 0.0, "mdd": 0.0, "sharpe": 0.0, "n_weeks": len(rets)}
    nav = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
    return {"cum": float(nav[-1] - 1), "mdd": mdd, "sharpe": sharpe, "n_weeks": len(rets)}


def main():
    with open(ADDED_DATES_PATH) as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}
    print(f"纳入日期记录: {len(added_dates)}只票")

    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    for label, price_cache_path, old_mom_cache_path in periods:
        print(f"\n{'='*20} {label} {'='*20}")
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]

        membership = compute_point_in_time_membership(c_df.index, list(c_df.columns), added_dates, {})

        # 量化一下这次过滤实际排除了多少"票-周"候选
        n_excluded_ticker_weeks = 0
        n_total_ticker_weeks = 0
        for date_str, valid in membership.items():
            n_total_ticker_weeks += len(c_df.columns)
            n_excluded_ticker_weeks += len(c_df.columns) - len(valid)
        print(f"  纳入日期过滤: 平均每周排除 {n_excluded_ticker_weeks/max(len(membership),1):.1f}/{len(c_df.columns)} 只票(还未被纳入标普500)")

        t0 = time.time()
        new_results = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=N_WORKERS, xgb_n_jobs=1,
            earnings_calendar=earnings_calendar,
            point_in_time_membership=membership,
        )
        print(f"  修正后回测耗时: {(time.time()-t0)/60:.1f}分钟")
        new_m = rough_metrics(new_results)

        with open(old_mom_cache_path, "rb") as f:
            old_results = pickle.load(f)
        old_m = rough_metrics(old_results)

        print(f"  修正前(未做纳入日期过滤，此前所有条目引用的数字): {old_m}")
        print(f"  修正后(纳入日期过滤生效): {new_m}")

        out_path = f"{SCRATCH}/pit_corrected_{label.replace('-','_')}_results.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(new_results, f)
        print(f"  修正后结果已存: {out_path}")


if __name__ == "__main__":
    main()
