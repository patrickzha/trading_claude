"""
Cycle3大修大补①：对"验证结论"第45条(动量+价值组合分散化收益)做正式的
block bootstrap显著性检验。这次样本量足够大(208周/291周)，不会重蹈
第42条"小样本全同号导致P=0%是假象"的陷阱——block bootstrap在236周量级
的样本上已经在这次会话里反复验证过是有效的检验工具(动量策略本身的
显著性检验一直用的就是这个方法)。

2022-2026段复用现成缓存(pit_full_run_with_trades.pkl)；2014-2020段
需要重新跑动量策略(~30分钟)，这次会把结果缓存下来供以后复用，不用
再跑一次。
"""
from __future__ import annotations

import pickle
import os

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
MOMENTUM_2022_CACHE = f"{SCRATCH}/pit_full_run_with_trades.pkl"
HIST_CACHE = f"{SCRATCH}/historical_2014_2020_cache.pkl"
MOMENTUM_2014_CACHE_OUT = f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"

BEST_WEIGHT = 0.5  # 两段区间交集附近的"稳健"权重，见第45条结论


def get_weekly_combo(mom_results, price_df, spy_close, membership_or_fake, value_weight):
    mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_df = mom_df.set_index("date")

    _, daily_nav = backtest_value_strategy(price_df, membership_or_fake, spy_close, rebalance_days=126, top_n=30)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")

    week_pairs = get_weekly_trading_dates(price_df.index)
    val_rows = []
    for monday, friday in week_pairs:
        try:
            nav_at_monday = nav_df.loc[:monday, "nav"].iloc[-1]
            nav_at_friday = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        val_rows.append({"date": monday, "value_ret": nav_at_friday / nav_at_monday - 1})
    val_weekly_ret = pd.DataFrame(val_rows).set_index("date")["value_ret"]

    merged = mom_df.join(val_weekly_ret, how="inner")
    blended = value_weight * merged["value_ret"] + (1 - value_weight) * merged["weekly_return"]
    return blended.values


def block_bootstrap_sharpe(rets, block=4, n_boot=5000, periods_per_year=52.0, seed=42):
    rng = np.random.default_rng(seed)
    rets = np.array(rets)
    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    boot_sharpes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([rets[i:i + block] for i in idx])[:n]
        s = sample.mean() / (sample.std() + 1e-9) * np.sqrt(periods_per_year)
        boot_sharpes.append(s)
    boot_sharpes = np.array(boot_sharpes)
    obs = rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)
    ci = np.percentile(boot_sharpes, [2.5, 97.5])
    p0 = float(np.mean(boot_sharpes <= 0))
    return obs, ci, p0


def main():
    with open(CACHE_PATH, "rb") as f:
        pit_cache = pickle.load(f)
    price_df_2022, spy_2022 = pit_cache["c"], pit_cache.get("spy")
    membership = compute_point_in_time_membership(price_df_2022.index, list(price_df_2022.columns),
                                                    pit_cache["added_dates"], pit_cache["removal_dates"])
    with open(MOMENTUM_2022_CACHE, "rb") as f:
        mom_2022 = pickle.load(f)

    combo_2022 = get_weekly_combo(mom_2022, price_df_2022, spy_2022, membership, BEST_WEIGHT)
    obs1, ci1, p01 = block_bootstrap_sharpe(combo_2022)
    print(f"2022-2026组合(value_weight={BEST_WEIGHT}): n={len(combo_2022)}周, Sharpe={obs1:.3f}, "
          f"95%CI=[{ci1[0]:.3f},{ci1[1]:.3f}], P(Sharpe<=0)={p01:.1%}")

    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = hist["c"], hist["o"], hist["h"], hist["l"], hist["v"], hist["vix"]
    hist_spy = hist.get("spy")
    sector_map = pit_cache["sector_map"]

    if os.path.exists(MOMENTUM_2014_CACHE_OUT):
        print("复用已缓存的2014-2020动量策略结果...")
        with open(MOMENTUM_2014_CACHE_OUT, "rb") as f:
            mom_2014 = pickle.load(f)
    else:
        print("跑2014-2020动量策略(约30分钟)...")
        n_workers = max(1, os.cpu_count() or 1)
        mom_2014 = run_walk_forward_backtest(
            c_df, o_df, h_df, l_df, v_df, vix,
            enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
            enable_earnings_blackout=False, sector_map=sector_map,
        )
        with open(MOMENTUM_2014_CACHE_OUT, "wb") as f:
            pickle.dump(mom_2014, f)
        print(f"已缓存到 {MOMENTUM_2014_CACHE_OUT}")

    all_tickers = list(c_df.columns)
    fake_membership = {c_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(c_df))}
    combo_2014 = get_weekly_combo(mom_2014, c_df, hist_spy, fake_membership, BEST_WEIGHT)
    obs2, ci2, p02 = block_bootstrap_sharpe(combo_2014)
    print(f"2014-2020组合(value_weight={BEST_WEIGHT}): n={len(combo_2014)}周, Sharpe={obs2:.3f}, "
          f"95%CI=[{ci2[0]:.3f},{ci2[1]:.3f}], P(Sharpe<=0)={p02:.1%}")


if __name__ == "__main__":
    main()
