"""
Cycle67小修：第299条备兑看涨期权(OTM5%/覆盖100%)的block bootstrap显著性
检验——这次会话对任何"点估计改善"的正面发现，标准流程都要求做显著性
检验才能真正采信(第267/268/276条建立的纪律)，第299条目前只有点估计，
补上这一步。直接对比"月度基线收益"vs"月度备兑收益"两个序列(逐月频率，
不折算成周)，避免额外的日频重建复杂度。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

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

BASE = dict(value_weighting="hrp", value_rebalance_days=5,
            value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

R_FREE = 0.04
MONTH_DAYS = 21


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def monthly_series(equity_nav: pd.Series, vix: pd.Series, otm_pct: float | None):
    """otm_pct=None时返回原始月度收益(基线)；否则返回叠加备兑看涨后的月度收益。"""
    idx = equity_nav.index
    n = len(idx)
    vix_aligned = vix.reindex(idx).ffill() / 100.0 if vix is not None else None
    rets = []
    i = 0
    while i + MONTH_DAYS < n:
        S0 = equity_nav.iloc[i]
        ST = equity_nav.iloc[i + MONTH_DAYS]
        raw_ret = ST / S0 - 1
        if otm_pct is None:
            rets.append(raw_ret)
        else:
            sigma = vix_aligned.iloc[i]
            if pd.isna(sigma) or sigma <= 0:
                sigma = 0.20
            K = S0 * (1 + otm_pct)
            T = MONTH_DAYS / 252.0
            premium = bs_call_price(S0, K, T, R_FREE, sigma)
            payoff = max(ST - K, 0.0)
            rets.append(raw_ret + premium / S0 - payoff / S0)
        i += MONTH_DAYS
    return np.array(rets)


def block_bootstrap_diff_pvalue(diff, block=3, n_boot=5000, seed=42):
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
    return float(diff.mean()), float(np.mean(boot_means <= 0))


for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    vix = cache.get("vix")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n{'='*70}\n {label}\n{'='*70}")
    r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
    equity_nav = r_base["full_nav"]

    if vix is None:
        print("  [警告] 无VIX数据，跳过")
        continue

    base_rets = monthly_series(equity_nav, vix, None)
    call_rets = monthly_series(equity_nav, vix, 0.05)
    diff = call_rets - base_rets
    mean_diff, p_le_zero = block_bootstrap_diff_pvalue(diff)
    print(f"  月收益差均值(备兑-基线)={mean_diff*100:.4f}%/月, P(差<=0)={p_le_zero:.3f}, n={len(diff)}个月")

print("\n全部完成。")
