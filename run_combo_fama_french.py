"""
Cycle3大修大补②：对动量+价值组合(value_weight=0.5)本身做Fama-French
三因子回归——分散化降低了残差方差，理论上有可能让原本在两条腿单独测
时都不显著的alpha，在组合后变得更接近显著（不是凭空创造alpha，是
降噪之后原本被噪声掩盖的小alpha更容易被测出来）。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
MOMENTUM_2022_CACHE = f"{SCRATCH}/pit_full_run_with_trades.pkl"
BEST_WEIGHT = 0.5


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
    return blended


def regress(weekly_ret: pd.Series, ff: pd.DataFrame, label: str):
    week_pairs_ff = []
    for date in weekly_ret.index:
        friday = date + pd.Timedelta(days=4)
        window = ff.loc[date:friday]
        if len(window) == 0:
            continue
        mkt = float((1 + window["Mkt-RF"]).prod() - 1)
        smb = float((1 + window["SMB"]).prod() - 1)
        hml = float((1 + window["HML"]).prod() - 1)
        rf = float((1 + window["RF"]).prod() - 1)
        week_pairs_ff.append({"date": date, "Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": rf})
    ff_weekly = pd.DataFrame(week_pairs_ff).set_index("date")

    merged = pd.DataFrame({"ret": weekly_ret}).join(ff_weekly, how="inner")
    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                          merged["SMB"].values, merged["HML"].values])
    beta, se, tvals = ols_alpha_beta(y, X)
    alpha_annualized = (1 + beta[0]) ** 52 - 1
    print(f"\n--- {label} (n={len(merged)}周) ---")
    for name, b, s, t in zip(["alpha", "Mkt-RF", "SMB", "HML"], beta, se, tvals):
        print(f"  {name}: 系数={b:.6f}, t值={t:.2f}")
    print(f"  年化alpha: {alpha_annualized*100:.2f}%, t值={tvals[0]:.2f} "
          f"({'显著' if abs(tvals[0]) > 2 else '不显著'})")


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()

    with open(CACHE_PATH, "rb") as f:
        pit_cache = pickle.load(f)
    price_df_2022, spy_2022 = pit_cache["c"], pit_cache.get("spy")
    membership = compute_point_in_time_membership(price_df_2022.index, list(price_df_2022.columns),
                                                    pit_cache["added_dates"], pit_cache["removal_dates"])
    with open(MOMENTUM_2022_CACHE, "rb") as f:
        mom_2022 = pickle.load(f)
    combo_2022 = get_weekly_combo(mom_2022, price_df_2022, spy_2022, membership, BEST_WEIGHT)
    regress(combo_2022, ff, "组合策略 2022-2026")

    with open(f"{SCRATCH}/historical_2014_2020_cache.pkl", "rb") as f:
        hist = pickle.load(f)
    c_df, spy_hist = hist["c"], hist.get("spy")
    with open(f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", "rb") as f:
        mom_2014 = pickle.load(f)
    all_tickers = list(c_df.columns)
    fake_membership = {c_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(c_df))}
    combo_2014 = get_weekly_combo(mom_2014, c_df, spy_hist, fake_membership, BEST_WEIGHT)
    regress(combo_2014, ff, "组合策略 2014-2020")


if __name__ == "__main__":
    main()
