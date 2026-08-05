"""
Cycle33小修小补(c)：第167条的Fama-French三因子alpha检验只测了2022-2026
一段作为代表，这次补上2014-2020/2009-2014两段，检验"四个权重方案alpha
都不显著、但beta_Mkt随框架系统性下降"这个结论是不是跨区间稳健的规律，
不是2022-2026这一段数据的偶然特征。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl"),
]


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()

    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    for label, price_cache_path in PERIODS:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label} ===")
        for scheme in ["equal", "min_variance", "risk_parity", "hrp"]:
            _, daily_nav = backtest_value_strategy(price_df, membership, spy_close, weighting=scheme)
            nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
            nav_df["ret"] = nav_df["nav"].pct_change()
            nav_df = nav_df.dropna()

            merged = nav_df.join(ff, how="inner")
            if len(merged) < 30:
                print(f"  {scheme}: 对齐后样本太少({len(merged)})，跳过")
                continue

            y = (merged["ret"] - merged["RF"]).values
            X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                                  merged["SMB"].values, merged["HML"].values])
            beta, se, tvals = ols_alpha_beta(y, X)
            alpha_annualized = (1 + beta[0]) ** 252 - 1
            print(f"  {scheme:14s}: 年化alpha={alpha_annualized*100:+.2f}%, alpha_t值={tvals[0]:.2f} "
                  f"({'显著' if abs(tvals[0]) > 2 else '不显著'}), "
                  f"beta_Mkt={beta[1]:.2f}, beta_SMB={beta[2]:.2f}, beta_HML={beta[3]:.2f}, n={len(merged)}")


if __name__ == "__main__":
    main()
