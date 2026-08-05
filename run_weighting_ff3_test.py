"""
Cycle32小修小补(b)：三个组合构建框架(min_variance/risk_parity/hrp)的价值腿
超额收益，此前(第140/155/159条)从未做过Fama-French三因子alpha显著性
检验——只有等权版本的价值策略基线测过FF3(见原始的run_fama_french_diagnostic.py，
README第38条附近)。这次把同样的FF3回归套到三个新权重方案上，检验"框架
改进"这部分超额收益本身是不是纯粹的因子暴露(alpha不显著=只是换了个
方式暴露同样的Mkt/SMB/HML因子)。用2022-2026这段(数据质量最高、point-in-time
成分股修正过)做代表性检验，不重复三段区间(FF3数据需要联网抓取，三段
X四个权重=12次网络请求，这里只测最新一段作为代表)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()
    print(f"FF3数据: {ff.index[0].date()} ~ {ff.index[-1].date()}, {len(ff)}天")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    spy_close = cache.get("spy")
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])

    for scheme in ["equal", "min_variance", "risk_parity", "hrp"]:
        print(f"\n跑价值策略(weighting={scheme})拿逐日净值...")
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
