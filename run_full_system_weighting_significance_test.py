"""
Cycle33小修小补(b)：第166条的block bootstrap显著性检验只测了价值腿单独
(不含动态债券对冲)，这次把同样的检验方法套到完整`full_system.py`层面
(第168条risk_parity、第142条min_variance、第160条hrp都已经在这个层面
验证过点估计的Sharpe/MaxDD改善)，检验"完整系统层面"的超额收益差异
是不是同样统计不显著——如果完整系统层面反而显著，说明动态债券对冲
可能放大了组合优化框架的价值；如果同样不显著，进一步巩固第166/167条
"风险管理而非新收益来源"这个判断。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]


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
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

        r_baseline = run_full_system(price_df, membership, spy_close, mom_results)
        wk_equal = to_weekly_rets(r_baseline["full_nav"])

        print(f"\n=== {label} ===")
        for scheme in ["min_variance", "risk_parity", "hrp"]:
            r_w = run_full_system(price_df, membership, spy_close, mom_results, value_weighting=scheme)
            wk_w = to_weekly_rets(r_w["full_nav"])
            common_idx = wk_equal.index.intersection(wk_w.index)
            diff = (wk_w.reindex(common_idx) - wk_equal.reindex(common_idx)).dropna()
            obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff.values)
            sig = "显著" if p_le_zero < 0.05 else ("边缘" if p_le_zero < 0.15 else "不显著")
            print(f"  {scheme:14s} vs 等权(full_system层面): 周均超额收益={obs_mean*100:+.4f}%, "
                  f"P(bootstrap均值<=0)={p_le_zero:.3f} [{sig}], n周={len(diff)}")


if __name__ == "__main__":
    main()
