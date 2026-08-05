"""
Cycle22大修大补①：第108-110条修正了动量腿在2014-2020/2009-2014的生存
偏差后，最需要重新检验的是执行摘要表格里直接引用的Fama-French alpha
显著性数字(2014-2020 t=2.49显著，2009-2014 t=1.50不显著)——这两个数字
是"三元组合到底有没有真正意义上的新edge"这个核心判断的统计依据，用的
是带偏差的动量数据算出来的，必须用修正后的数据重新做一次。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl",
     f"{SCRATCH}/pit_corrected_2014_2020_results.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl",
     f"{SCRATCH}/pit_corrected_2009_2014_results.pkl"),
]

print("拉取Fama-French三因子日数据...")
ff = fetch_ff3_daily()
print(f"FF3数据: {ff.index[0].date()} ~ {ff.index[-1].date()}, {len(ff)}天")

for label, price_cache_path, old_mom_path, new_mom_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    with open(old_mom_path, "rb") as f:
        old_mom = pickle.load(f)
    with open(new_mom_path, "rb") as f:
        new_mom = pickle.load(f)

    print(f"\n{'='*20} {label} {'='*20}")
    for tag, mom in [("修正前(此前引用的数字)", old_mom), ("修正后(PIT修正)", new_mom)]:
        report = run_combo_strategy(price_df, membership, spy_close, mom)
        combo_nav = report["combo_nav"]
        nav_df = pd.DataFrame({"nav": combo_nav})
        nav_df["ret"] = nav_df["nav"].pct_change()
        nav_df = nav_df.dropna()

        merged = nav_df.join(ff, how="inner")
        if len(merged) < 30:
            print(f"  {tag}: 对齐后样本太少")
            continue

        y = (merged["ret"] - merged["RF"]).values
        X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                             merged["SMB"].values, merged["HML"].values])
        beta, se, tvals = ols_alpha_beta(y, X)
        alpha_annualized = (1 + beta[0]) ** 252 - 1
        sig = "显著(|t|>2)" if abs(tvals[0]) > 2 else "不显著(|t|<=2)"
        print(f"  {tag}: 年化alpha={alpha_annualized*100:.2f}%, alpha t值={tvals[0]:.2f} ({sig}), "
              f"beta_Mkt={beta[1]:.2f}, beta_SMB={beta[2]:.2f}, beta_HML={beta[3]:.2f}")
