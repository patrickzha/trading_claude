"""
Cycle32小修小补(a)：三个组合构建框架(min_variance/risk_parity/hrp)相对
等权的Sharpe改善，此前(第140/155/159/163条)只报告了点估计数字，从未
做过显著性检验——不知道这些改善是真实信号还是同一堆收益重新分配后
的噪声。用block bootstrap直接检验"加权版本 - 等权版本"这个每周收益率
差值序列的均值是否显著大于0(块长4，呼应周收益的自相关跨度，跟
stats_utils.block_bootstrap_sharpe同样的方法论，这里检验的是收益率差值
均值而不是Sharpe，因为我们关心的问题是"这个改进本身是不是噪声"，不是
"这个策略本身有没有正收益"，两者是不同的假设检验)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]


def to_weekly_rets(daily_nav):
    s = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
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

    for label, price_cache_path, is_pit in PERIODS:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label} ===")
        _, nav_equal = backtest_value_strategy(price_df, membership, spy_close, weighting="equal")
        wk_equal = to_weekly_rets(nav_equal)

        for scheme in ["min_variance", "risk_parity", "hrp"]:
            _, nav_w = backtest_value_strategy(price_df, membership, spy_close, weighting=scheme)
            wk_w = to_weekly_rets(nav_w)
            common_idx = wk_equal.index.intersection(wk_w.index)
            diff = (wk_w.reindex(common_idx) - wk_equal.reindex(common_idx)).dropna()
            obs_mean, p_le_zero = block_bootstrap_diff_pvalue(diff.values)
            sig = "显著" if p_le_zero < 0.05 else ("边缘" if p_le_zero < 0.15 else "不显著")
            print(f"  {scheme:14s} vs 等权: 周均超额收益={obs_mean*100:+.4f}%, "
                  f"P(bootstrap均值<=0)={p_le_zero:.3f} [{sig}], n周={len(diff)}")


if __name__ == "__main__":
    main()
