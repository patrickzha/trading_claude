"""
Cycle5小修小补：第45/46/47/51/53条的组合策略测试，价值腿用的都是半年度
(126天)调仓，但第48条已经发现半年度调仓的Sharpe因为观测点太少被高估。
这次用第48条更可信的月度(21天)调仓重新算一遍三段区间的组合bootstrap+
FF3检验，确认之前的"2/3显著(bootstrap)、1/3显著(FF3)"这个结论在换成
更细颗粒度后还成不成立。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
BEST_WEIGHT = 0.5


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


def get_weekly_combo(mom_results, price_df, spy_close, membership_or_fake, value_weight, rebalance_days=21):
    mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_df = mom_df.set_index("date")

    _, daily_nav = backtest_value_strategy(price_df, membership_or_fake, spy_close,
                                           rebalance_days=rebalance_days, top_n=30)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")

    week_pairs = get_weekly_trading_dates(price_df.index)
    val_rows = []
    for monday, friday in week_pairs:
        try:
            a = nav_df.loc[:monday, "nav"].iloc[-1]
            b = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        val_rows.append({"date": monday, "value_ret": b / a - 1})
    val_weekly_ret = pd.DataFrame(val_rows).set_index("date")["value_ret"]
    merged = mom_df.join(val_weekly_ret, how="inner")
    return value_weight * merged["value_ret"] + (1 - value_weight) * merged["weekly_return"]


def regress_ff3(weekly_ret, ff, label):
    rows = []
    for date in weekly_ret.index:
        friday = date + pd.Timedelta(days=4)
        window = ff.loc[date:friday]
        if len(window) == 0:
            continue
        row = {"date": date}
        for col in ["Mkt-RF", "SMB", "HML", "RF"]:
            row[col] = float((1 + window[col]).prod() - 1)
        rows.append(row)
    ff_weekly = pd.DataFrame(rows).set_index("date")
    merged = pd.DataFrame({"ret": weekly_ret}).join(ff_weekly, how="inner")
    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged))] + [merged[c].values for c in ["Mkt-RF", "SMB", "HML"]])
    beta, se, tvals = ols_alpha_beta(y, X)
    alpha_annualized = (1 + beta[0]) ** 52 - 1
    print(f"  [{label}] alpha年化={alpha_annualized*100:.2f}%, t={tvals[0]:.2f} "
          f"({'显著' if abs(tvals[0]) > 2 else '不显著'})")


def main():
    ff = fetch_ff3_daily()

    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", f"{SCRATCH}/pit_full_run_with_trades.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
    ]

    for label, price_cache_path, mom_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
        with open(mom_cache_path, "rb") as f:
            mom_results = pickle.load(f)

        combo = get_weekly_combo(mom_results, price_df, spy_close, membership, BEST_WEIGHT, rebalance_days=21)
        obs, ci, p0 = block_bootstrap_sharpe(combo.values)
        print(f"\n=== {label} (月度调仓价值腿, n={len(combo)}周) ===")
        print(f"  bootstrap: Sharpe={obs:.3f}, 95%CI=[{ci[0]:.3f},{ci[1]:.3f}], P(<=0)={p0:.1%}")
        regress_ff3(combo, ff, label)


if __name__ == "__main__":
    main()
