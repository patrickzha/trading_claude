"""
Cycle62大修大补①：低波动腿止损阈值优化，用第276/280条建立的完整三步流程
(点估计网格→触发比例/偏度分布诊断→block bootstrap显著性检验)，一次性
做完不分开跑——value腿(第276条)已经验证过3%是比默认5%更优的阈值，但
低波动腿的止损阈值从第234/235条设定5%以来从未用同样严格的方法重新
优化过，这次补上，保持两条腿的验证深度一致。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import skew

from full_system import run_full_system
from low_vol_investing import select_low_vol_portfolio
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

BASE = dict(value_weighting="hrp", value_rebalance_days=5, value_stop_loss_pct=0.03)


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
    return float(np.mean(boot_means <= 0))


print(f"\n{'='*70}\n 步骤1: 低波动腿止损网格(2%-6%，含默认5%)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for lv_sl in [0.02, 0.03, 0.04, 0.05, 0.06]:
        r = run_full_system(price_df, membership, spy_close, mom_results,
                             low_vol_stop_loss_pct=lv_sl, **BASE)
        print(f"  low_vol_stop_loss_pct={lv_sl}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 步骤2: 分布诊断(触发比例/偏度)——低波动腿5天调仓、HRP权重不适用(低波动腿一直等权)\n{'='*70}")


def instrumented_lowvol_backtest(price_df, membership, stop_loss_pct, rebalance_days=21):
    index = price_df.index
    n = len(index)
    rebalance_idx = 252 + 60
    n_triggered = 0
    n_total_positions = 0
    period_rets = []
    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]
        picks = [p for p in select_low_vol_portfolio(price_df, valid_tickers, rebalance_date, top_n=30) if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue
        weights = pd.Series(1.0 / len(picks), index=picks)
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks
                         if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        stopped_out = {}
        day_vals = []
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    w = weights.get(t, 0)
                    if t in stopped_out:
                        day_val += w * stopped_out[t]
                        total_w += w
                        continue
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        factor = p / p0
                        if factor <= (1 - stop_loss_pct):
                            factor = 1 - stop_loss_pct
                            stopped_out[t] = factor
                        day_val += w * factor
                        total_w += w
            if total_w > 0:
                day_vals.append(day_val / total_w)
        n_total_positions += len(entry_prices)
        n_triggered += len(stopped_out)
        if day_vals:
            period_rets.append(day_vals[-1] - 1.0)
        rebalance_idx += rebalance_days
    return np.array(period_rets), n_triggered, n_total_positions


for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for pct in [0.05, 0.03]:
        rets, n_trig, n_pos = instrumented_lowvol_backtest(price_df, membership, pct)
        trig_rate = n_trig / n_pos * 100 if n_pos else 0.0
        print(f"  止损{pct*100:.0f}%: 触发比例={trig_rate:.1f}%, 偏度={skew(rets):.3f}, "
              f"单期最大亏损={rets.min()*100:.2f}%")

print(f"\n{'='*70}\n 步骤3: 若3%通过诊断，做block bootstrap显著性检验(3% vs 5%)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    r_base = run_full_system(price_df, membership, spy_close, mom_results, low_vol_stop_loss_pct=0.05, **BASE)
    r_new = run_full_system(price_df, membership, spy_close, mom_results, low_vol_stop_loss_pct=0.03, **BASE)
    w_base = to_weekly(r_base["full_nav"], price_df)
    w_new = to_weekly(r_new["full_nav"], price_df)
    common = w_base.index.intersection(w_new.index)
    diff = (w_new.loc[common] - w_base.loc[common]).to_numpy(dtype=float)
    p_le_zero = block_bootstrap_diff_pvalue(diff)
    print(f"  低波动3% vs 5%: Sharpe {r_base['Sharpe']:.3f}->{r_new['Sharpe']:.3f}, "
          f"MaxDD {r_base['MaxDD']*100:.2f}%->{r_new['MaxDD']*100:.2f}%, P(差<=0)={p_le_zero:.3f}")

print("\n全部完成。")
