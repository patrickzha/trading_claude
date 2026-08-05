"""
Cycle35小修小补(c)：第170条截面复合评分组合在2014-2020/2009-2014两段
明显落后于现有三元组合(第55条)，这次用block bootstrap检验这个差距是否
统计显著——如果本身就不显著，说明"复合评分更差"这个判断的置信度需要
打折扣，不能把点估计的差距直接当成确定性结论。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from run_cross_sectional_composite_test import backtest_composite
from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def to_weekly_rets(nav_series_or_list):
    if isinstance(nav_series_or_list, list):
        s = pd.DataFrame(nav_series_or_list, columns=["date", "nav"]).set_index("date")["nav"]
    else:
        s = nav_series_or_list
    weekly = s.resample("W-FRI").last().dropna()
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

        composite_nav_list = backtest_composite(price_df, membership)
        combo_report = run_combo_strategy(price_df, membership, spy_close, mom_results)

        wk_composite = to_weekly_rets(composite_nav_list)
        wk_combo = to_weekly_rets(combo_report["combo_nav"])
        common_idx = wk_composite.index.intersection(wk_combo.index)
        diff = (wk_composite.reindex(common_idx) - wk_combo.reindex(common_idx)).dropna()
        obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff.values)
        sig = "显著" if p_le_zero < 0.05 else ("边缘" if p_le_zero < 0.15 else "不显著")
        direction = "复合评分更好" if obs_mean > 0 else "现有三元组合更好"
        print(f"\n=== {label} ===")
        print(f"  复合评分 vs 现有三元组合: 周均差={obs_mean*100:+.4f}%({direction}), "
              f"P(bootstrap均值方向翻转)={p_le_zero:.3f} [{sig}], n周={len(diff)}")


if __name__ == "__main__":
    main()
