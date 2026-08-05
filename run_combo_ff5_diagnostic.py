"""
Cycle5大修大补①：Fama-French五因子(加RMW盈利能力因子、CMA投资因子)回归。
"验证结论"第47/51条只测过三因子(Mkt-RF/SMB/HML)，这次补上RMW/CMA两个
因子，看组合策略的alpha是不是被这两个额外因子进一步解释掉——如果RMW/
CMA暴露显著，说明这个价值策略选出来的"高ROE"股票同时也在吃盈利能力
因子的溢价，alpha的"新增"部分应该更小；如果不显著，前三因子的结论
不受影响。三段区间都测，复用已缓存的动量策略结果，不用重新跑。
"""
from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests

from backtest import get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from run_fama_french_diagnostic import ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
BEST_WEIGHT = 0.5


def fetch_ff5_daily():
    r = requests.get(FF5_URL, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(z.namelist()[0]) as f:
        content = f.read().decode("latin1")
    lines = content.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    end = next((i for i, l in enumerate(lines) if i > start and l.strip() == ""), len(lines))
    rows = []
    for l in lines[start + 1:end]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != 7 or not parts[0].isdigit():
            continue
        date = pd.Timestamp(parts[0])
        rows.append([date] + [float(x) / 100.0 for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]).set_index("date")
    return df


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
            a = nav_df.loc[:monday, "nav"].iloc[-1]
            b = nav_df.loc[:friday, "nav"].iloc[-1]
        except IndexError:
            continue
        val_rows.append({"date": monday, "value_ret": b / a - 1})
    val_weekly_ret = pd.DataFrame(val_rows).set_index("date")["value_ret"]

    merged = mom_df.join(val_weekly_ret, how="inner")
    return (value_weight * merged["value_ret"] + (1 - value_weight) * merged["weekly_return"])


def regress_ff5(weekly_ret: pd.Series, ff: pd.DataFrame, label: str):
    rows = []
    for date in weekly_ret.index:
        friday = date + pd.Timedelta(days=4)
        window = ff.loc[date:friday]
        if len(window) == 0:
            continue
        row = {"date": date}
        for col in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]:
            row[col] = float((1 + window[col]).prod() - 1)
        rows.append(row)
    ff_weekly = pd.DataFrame(rows).set_index("date")
    merged = pd.DataFrame({"ret": weekly_ret}).join(ff_weekly, how="inner")
    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged))] + [merged[c].values for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]])
    beta, se, tvals = ols_alpha_beta(y, X)
    alpha_annualized = (1 + beta[0]) ** 52 - 1
    print(f"\n--- {label} FF5回归 (n={len(merged)}周) ---")
    for name, b, t in zip(["alpha", "Mkt-RF", "SMB", "HML", "RMW", "CMA"], beta, tvals):
        print(f"  {name}: 系数={b:.6f}, t值={t:.2f}")
    print(f"  年化alpha: {alpha_annualized*100:.2f}%, t值={tvals[0]:.2f} "
          f"({'显著' if abs(tvals[0]) > 2 else '不显著'})")


def main():
    print("拉取Fama-French五因子日数据...")
    ff5 = fetch_ff5_daily()
    print(f"FF5数据: {ff5.index[0].date()} ~ {ff5.index[-1].date()}")

    with open(f"{SCRATCH}/point_in_time_universe_cache.pkl", "rb") as f:
        pit_cache = pickle.load(f)
    price_df_2022, spy_2022 = pit_cache["c"], pit_cache.get("spy")
    membership = compute_point_in_time_membership(price_df_2022.index, list(price_df_2022.columns),
                                                    pit_cache["added_dates"], pit_cache["removal_dates"])
    with open(f"{SCRATCH}/pit_full_run_with_trades.pkl", "rb") as f:
        mom_2022 = pickle.load(f)
    combo_2022 = get_weekly_combo(mom_2022, price_df_2022, spy_2022, membership, BEST_WEIGHT)
    regress_ff5(combo_2022, ff5, "组合策略 2022-2026")

    with open(f"{SCRATCH}/historical_2014_2020_cache.pkl", "rb") as f:
        hist2014 = pickle.load(f)
    c_2014, spy_2014 = hist2014["c"], hist2014.get("spy")
    with open(f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", "rb") as f:
        mom_2014 = pickle.load(f)
    fake_2014 = {c_2014.index[i].strftime("%Y-%m-%d"): list(c_2014.columns) for i in range(len(c_2014))}
    combo_2014 = get_weekly_combo(mom_2014, c_2014, spy_2014, fake_2014, BEST_WEIGHT)
    regress_ff5(combo_2014, ff5, "组合策略 2014-2020")

    with open(f"{SCRATCH}/historical_2009_2014_cache.pkl", "rb") as f:
        hist2009 = pickle.load(f)
    c_2009, spy_2009 = hist2009["c"], hist2009.get("spy")
    with open(f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", "rb") as f:
        mom_2009 = pickle.load(f)
    fake_2009 = {c_2009.index[i].strftime("%Y-%m-%d"): list(c_2009.columns) for i in range(len(c_2009))}
    combo_2009 = get_weekly_combo(mom_2009, c_2009, spy_2009, fake_2009, BEST_WEIGHT)
    regress_ff5(combo_2009, ff5, "组合策略 2009-2014")


if __name__ == "__main__":
    main()
