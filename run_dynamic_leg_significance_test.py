"""
Cycle34小修小补(d)：第177条动态因子腿配置在2014-2020这段MaxDD改善幅度
很大(-28.75%->-16.42%)，但三段区间方向不一致(2022-2026更差)，这次用
第166条同样的block bootstrap方法论检验2014-2020这段的超额收益是不是
统计显著，而不是"看起来很好看但其实是噪声"——如实检验这个最亮眼的
数字是否经得起显著性检验，避免选择性地把它当成确定性结论。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from run_dynamic_leg_allocation_test import dynamic_weight_combo
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def to_weekly_rets(nav_series: pd.Series):
    weekly = nav_series.resample("W-FRI").last().dropna()
    return weekly.pct_change().dropna()


def block_bootstrap_diff_pvalue(diff, block=4, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        sample = np.concatenate([diff[i:i + block] for i in idx])[:n]
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    obs_mean = diff.mean()
    p_le_zero = float(np.mean(boot_means <= 0))
    return obs_mean, p_le_zero


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
         f"{SCRATCH}/default_full_results_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
         f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
         f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
    ]

    for label, price_cache_path, mom_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        report = run_combo_strategy(price_df, membership, spy_close, mom_results)
        leg_nav = report["leg_nav"]
        equal_nav = report["combo_nav"]
        dyn_nav = dynamic_weight_combo(leg_nav)

        wk_equal = to_weekly_rets(equal_nav)
        wk_dyn = to_weekly_rets(dyn_nav)
        common_idx = wk_equal.index.intersection(wk_dyn.index)
        diff = (wk_dyn.reindex(common_idx) - wk_equal.reindex(common_idx)).dropna()
        obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff.values)
        sig = "显著" if p_le_zero < 0.05 else ("边缘" if p_le_zero < 0.15 else "不显著")
        print(f"\n=== {label} ===")
        print(f"  动态加权 vs 固定等权: 周均超额收益={obs_mean*100:+.4f}%, P(bootstrap均值<=0)={p_le_zero:.3f} [{sig}], n周={len(diff)}")


if __name__ == "__main__":
    main()
