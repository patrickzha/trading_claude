"""
Cycle59小修小补(a)(b)(c)(d)合并：补全第272条(黄金分散化叠加止损+HRP配置)
留下的验证缺口。

(a) 把gold_split测试扩展到2009-2014/2014-2020两段区间(第272条只测了
    2022-2026)，第84/85/120条已经确认黄金在2009-2014这段disinflationary
    复苏期明显跑输债券，这里直接验证同样的权衡关系在新的止损+HRP底座下
    是否依然成立。
(b) 用block bootstrap检验第272条2022-2026段gold_split=0.3 vs 0(基线)
    的收益率差异是否统计显著，不只看点估计Sharpe。
(c) dollar_split(第206/207/213条，已确认在2022-2026正贡献但同样可能是
    区间依赖)套进同一个止损+HRP底座做同样的三段测试，跟gold_split做
    平行对照。
(d) value_stop_loss_pct/low_vol_stop_loss_pct不再假设两条腿最优阈值
    相同，测试非对称组合(3%/5%、5%/3%)是否优于当前默认的对称5%/5%。
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

BASE_KW = dict(value_weighting="hrp", value_rebalance_days=5,
               value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.05)


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


print(f"\n{'='*70}\n (a) gold_split三段区间完整测试(止损+HRP底座)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for gs in [0.0, 0.3, 0.5, 0.7]:
        r = run_full_system(price_df, membership, spy_close, mom_results, gold_split=gs, **BASE_KW)
        print(f"  gold_split={gs}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%, CAGR={r['CAGR']*100:.2f}%")

print(f"\n{'='*70}\n (b) 2022-2026段 gold_split=0.3 vs 0 显著性检验(block bootstrap)\n{'='*70}")
price_df, spy_close, mom_results, membership = load_period(*PERIODS[0][1:])
r0 = run_full_system(price_df, membership, spy_close, mom_results, gold_split=0.0, **BASE_KW)
r3 = run_full_system(price_df, membership, spy_close, mom_results, gold_split=0.3, **BASE_KW)
w0 = to_weekly(r0["full_nav"], price_df)
w3 = to_weekly(r3["full_nav"], price_df)
common = w0.index.intersection(w3.index)
diff = (w3.loc[common] - w0.loc[common]).to_numpy(dtype=float)
obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff)
print(f"  周收益差均值(gold0.3-gold0)={obs_mean*100:.4f}%/周, P(差<=0)={p_le_zero:.3f}, n={len(diff)}周")

print(f"\n{'='*70}\n (c) dollar_split三段区间平行测试(止损+HRP底座)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for ds in [0.0, 0.2, 0.3]:
        r = run_full_system(price_df, membership, spy_close, mom_results, dollar_split=ds, **BASE_KW)
        print(f"  dollar_split={ds}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%, CAGR={r['CAGR']*100:.2f}%")

print(f"\n{'='*70}\n (d) value/low_vol止损阈值非对称组合测试(三段区间)\n{'='*70}")
ASYM_CONFIGS = {
    "对称5%/5%(当前默认)": dict(value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.05),
    "value3%/lowvol5%": dict(value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05),
    "value5%/lowvol3%": dict(value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.03),
}
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for cname, kwargs in ASYM_CONFIGS.items():
        r = run_full_system(price_df, membership, spy_close, mom_results,
                             value_weighting="hrp", value_rebalance_days=5, **kwargs)
        print(f"  {cname}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%, CAGR={r['CAGR']*100:.2f}%")

print("\n全部完成。")
