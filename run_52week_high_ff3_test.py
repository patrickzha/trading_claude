"""
Cycle41小修小补(b)：52周高点因子(第216条)的Fama-French三因子alpha检验——
第216条只报告了CAGR/Sharpe/MaxDD超额收益，这次检验负超额收益是否在
因子暴露之外有统计显著性，还是只是噪声。用2022-2026代表性区间。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from run_52week_high_factor_test import backtest_52w_high
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])

    print("跑52周高点因子策略拿逐日净值...")
    daily_nav = backtest_52w_high(price_df, membership)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
    nav_df["ret"] = nav_df["nav"].pct_change()
    nav_df = nav_df.dropna()

    merged = nav_df.join(ff, how="inner")
    print(f"对齐后共 {len(merged)} 个交易日")
    if len(merged) < 30:
        print("对齐后样本太少，无法做有意义的回归")
        return

    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                          merged["SMB"].values, merged["HML"].values])
    beta, se, tvals = ols_alpha_beta(y, X)
    alpha_annualized = (1 + beta[0]) ** 252 - 1

    print(f"\n52周高点因子: 年化alpha={alpha_annualized*100:+.2f}%, alpha_t值={tvals[0]:.2f} "
          f"({'显著' if abs(tvals[0]) > 2 else '不显著'}), "
          f"beta_Mkt={beta[1]:.2f}, beta_SMB={beta[2]:.2f}, beta_HML={beta[3]:.2f}, n={len(merged)}")


if __name__ == "__main__":
    main()
