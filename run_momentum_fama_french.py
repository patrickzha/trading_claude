"""
Cycle2大修大补②：把最初那套周频动量策略也放进Fama-French框架回归一遍，
跟"验证结论"第39条价值策略的处理方式完全对称——用同一把尺子比较两个
策略，看动量策略的历史表现有多少是已知因子(市场beta/规模/价值)解释
得了的，有多少是因子之外真正的alpha。

动量策略是周频调仓(不是每日)，FF3每日因子需要先转成周频再对齐，用
周一到周五这个跟策略完全一致的周期算周收益率。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH, EARNINGS_CACHE
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map = cache["sector_map"]
    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    with open(EARNINGS_CACHE, "rb") as f:
        earnings_calendar = pickle.load(f)

    import os
    n_workers = max(1, os.cpu_count() or 1)
    print("跑动量策略(默认配置)拿逐周收益率...")
    results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        earnings_calendar=earnings_calendar, sector_map=sector_map,
        point_in_time_membership=membership,
    )

    # 把FF3日因子聚合成周频(周一到周五，跟策略的调仓周期完全对齐)
    week_pairs = get_weekly_trading_dates(c_df.index)
    week_ff = []
    for monday, friday in week_pairs:
        window = ff.loc[monday:friday]
        if len(window) == 0:
            continue
        # 周收益率=每日简单收益率的复利乘积-1(不是简单加总，更准确)
        mkt = float((1 + window["Mkt-RF"]).prod() - 1)
        smb = float((1 + window["SMB"]).prod() - 1)
        hml = float((1 + window["HML"]).prod() - 1)
        rf = float((1 + window["RF"]).prod() - 1)
        week_ff.append({"date": monday.strftime("%Y-%m-%d"), "Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": rf})
    ff_weekly = pd.DataFrame(week_ff).set_index("date")

    strat = pd.DataFrame(results)[["date", "weekly_return"]].set_index("date")
    merged = strat.join(ff_weekly, how="inner")
    print(f"对齐后共 {len(merged)} 周")

    y = (merged["weekly_return"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                          merged["SMB"].values, merged["HML"].values])
    beta, se, tvals = ols_alpha_beta(y, X)
    names = ["alpha(周)", "beta_Mkt-RF", "beta_SMB", "beta_HML"]

    print("\n" + "=" * 60)
    print(" Fama-French三因子回归结果(周频动量策略超额收益)")
    print("=" * 60)
    for name, b, s, t in zip(names, beta, se, tvals):
        print(f"  {name}: 系数={b:.6f}, 标准误={s:.6f}, t值={t:.2f}")
    alpha_annualized = (1 + beta[0]) ** 52 - 1
    print(f"\n  年化alpha: {alpha_annualized*100:.2f}%")
    print(f"  alpha的t值: {tvals[0]:.2f} ({'显著(|t|>2)' if abs(tvals[0]) > 2 else '不显著(|t|<=2)'})")


if __name__ == "__main__":
    main()
