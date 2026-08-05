"""
Cycle2小修小补：价值策略的block bootstrap显著性检验。样本量天然小
(7-22个半年度/季度独立持有期)，正式做一次跟动量策略同样标准的检验，
而不是只看CAGR数字好看就当真。

用period_ret序列(不是daily_nav，因为调仓period本身就是天然的"块"，
不需要再额外分块)做bootstrap重采样，算Sharpe的置信区间。
"""
from __future__ import annotations

import pickle

import numpy as np

from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

PIT_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"
HIST_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"


def bootstrap_sharpe_ci(rets, periods_per_year, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    rets = np.array(rets)
    n = len(rets)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = rng.choice(rets, size=n, replace=True)
        s = sample.mean() / (sample.std() + 1e-9) * np.sqrt(periods_per_year)
        boot_sharpes.append(s)
    boot_sharpes = np.array(boot_sharpes)
    obs_sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)
    ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
    p_le_zero = float(np.mean(boot_sharpes <= 0))
    return obs_sharpe, ci_low, ci_high, p_le_zero


def main():
    with open(PIT_CACHE, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])
    results, _ = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=126, top_n=30)
    rets = [r["period_ret"] for r in results]
    obs, lo, hi, p0 = bootstrap_sharpe_ci(rets, periods_per_year=2.0)
    print(f"点位时间正确554只(2022-2026, n={len(rets)}个半年度观测): "
          f"Sharpe={obs:.3f}, 95%CI=[{lo:.3f}, {hi:.3f}], P(Sharpe<=0)={p0:.1%}")

    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    hist_price_df, hist_spy = hist["c"], hist.get("spy")
    all_tickers = list(hist_price_df.columns)
    fake_membership = {hist_price_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(hist_price_df))}
    results2, _ = backtest_value_strategy(hist_price_df, fake_membership, hist_spy, rebalance_days=126, top_n=30)
    rets2 = [r["period_ret"] for r in results2]
    obs2, lo2, hi2, p02 = bootstrap_sharpe_ci(rets2, periods_per_year=2.0)
    print(f"独立历史区间2014-2020(n={len(rets2)}个半年度观测): "
          f"Sharpe={obs2:.3f}, 95%CI=[{lo2:.3f}, {hi2:.3f}], P(Sharpe<=0)={p02:.1%}")

    # 两段拼起来做一次合并检验(方向性参考，两段市场机制不同，只是增加样本量的近似处理)
    combined = rets + rets2
    obs3, lo3, hi3, p03 = bootstrap_sharpe_ci(combined, periods_per_year=2.0)
    print(f"两段合并(n={len(combined)}个观测，注意跨机制拼接是近似处理): "
          f"Sharpe={obs3:.3f}, 95%CI=[{lo3:.3f}, {hi3:.3f}], P(Sharpe<=0)={p03:.1%}")


if __name__ == "__main__":
    main()
