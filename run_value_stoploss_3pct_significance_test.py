"""
Cycle59后续：value3%/lowvol5%非对称止损组合(小修(d)的初步发现)三段区间点估计
Sharpe/MaxDD都比对称5%/5%好，但诊断脚本(run_value_stoploss_3pct_asymmetric_
distribution_check.py)显示3%阈值下触发比例翻倍(9.6-21.8%区间)、偏度在三段
区间都上升，且单期最大亏损明显被"钉"在3%附近——是第239/245/267/268条
反复警示过的"更紧止损→伪凸性"这类可疑信号的中等强度版本，不是极端案例
(远低于第239条67%触发比例那种一眼假的情形)，需要用第267/268条同样的
block bootstrap方法直接检验Sharpe改善是不是统计噪声，不能只看点估计。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from stats_utils import to_weekly

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]


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


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership


for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    print(f"\n{'='*70}\n {label}\n{'='*70}")
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)

    r_base = run_full_system(price_df, membership, spy_close, mom_results,
                              value_weighting="hrp", value_rebalance_days=5,
                              value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.05)
    r_asym = run_full_system(price_df, membership, spy_close, mom_results,
                              value_weighting="hrp", value_rebalance_days=5,
                              value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

    print(f"  对称5%/5%: Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%")
    print(f"  value3%/lowvol5%: Sharpe={r_asym['Sharpe']:.3f}, MaxDD={r_asym['MaxDD']*100:.2f}%")

    w_base = to_weekly(r_base["full_nav"], price_df)
    w_asym = to_weekly(r_asym["full_nav"], price_df)
    common = w_base.index.intersection(w_asym.index)
    diff = (w_asym.loc[common] - w_base.loc[common]).to_numpy(dtype=float)
    obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff)
    print(f"  周收益差均值(3%-5%)={obs_mean*100:.4f}%/周, P(差<=0)={p_le_zero:.3f}, n={len(diff)}周")

print("\n全部完成。")
