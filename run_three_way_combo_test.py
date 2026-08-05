"""
Cycle6大修大补：三元组合(动量+价值+低波动)测试。低波动因子跟动量、价值
机制都不同，理论上进一步降相关性。三段区间都测，动量/价值用已缓存结果，
低波动策略新跑(便宜，纯价格数据，不涉及模型训练)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from low_vol_investing import backtest_low_vol_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


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


def to_weekly(daily_nav, price_df):
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
    week_pairs = get_weekly_trading_dates(price_df.index)
    rows = []
    for monday, friday in week_pairs:
        try:
            a = nav_df.loc[:monday, "nav"].iloc[-1]
            b = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        rows.append({"date": monday, "ret": b / a - 1})
    return pd.DataFrame(rows).set_index("date")["ret"]


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
        mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
        mom_df["date"] = pd.to_datetime(mom_df["date"])
        mom_ret = mom_df.set_index("date")["weekly_return"]

        _, val_daily_nav = backtest_value_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
        val_ret = to_weekly(val_daily_nav, price_df)

        _, lv_daily_nav = backtest_low_vol_strategy(price_df, membership, spy_close, rebalance_days=21, top_n=30)
        lv_ret = to_weekly(lv_daily_nav, price_df)

        merged = pd.DataFrame({"mom": mom_ret, "val": val_ret, "lv": lv_ret}).dropna()
        print(f"\n=== {label} (n={len(merged)}周) ===")
        print(f"  相关系数矩阵:\n{merged.corr().round(3)}")

        equal_third = merged.mean(axis=1)
        obs, ci, p0 = block_bootstrap_sharpe(equal_third.values)
        print(f"  三元等权组合: Sharpe={obs:.3f}, 95%CI=[{ci[0]:.3f},{ci[1]:.3f}], P(<=0)={p0:.1%}")
        regress_ff3(equal_third, ff, f"{label} 三元等权")


if __name__ == "__main__":
    main()
