"""
Cycle4大修大补②：在第三段完全独立的历史区间(2009-2014)上，把动量策略、
价值策略、组合策略(value_weight=0.5)全套跑一遍，包括block bootstrap和
Fama-French回归——直接检验"验证结论"第47条的悬念：2014-2020显著的
组合alpha，是真实规律还是那一段区间本身的巧合？这是判定性的第三局。

已知局限：跟"验证结论"第17条同样，2009-2014这份缓存用当前484只成分股
名单去跑历史，没有点位时间正确的成分股过滤，方向性结论可信、绝对数字
可能偏乐观。另外这段区间SEC基本面数据覆盖率可能不如后面几段完整
（多数票从2010-2011年才有披露记录），价值策略在2009-2010这段的候选池
会比正常情况稀疏。
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, get_weekly_trading_dates
from value_investing import backtest_value_strategy
from run_fama_french_diagnostic import fetch_ff3_daily, ols_alpha_beta

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
CACHE_PATH = f"{SCRATCH}/historical_2009_2014_cache.pkl"
PIT_CACHE = f"{SCRATCH}/point_in_time_universe_cache.pkl"
MOM_OUT = f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"

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


def regress_ff3(weekly_ret: pd.Series, ff: pd.DataFrame, label: str):
    rows = []
    for date in weekly_ret.index:
        friday = date + pd.Timedelta(days=4)
        window = ff.loc[date:friday]
        if len(window) == 0:
            continue
        mkt = float((1 + window["Mkt-RF"]).prod() - 1)
        smb = float((1 + window["SMB"]).prod() - 1)
        hml = float((1 + window["HML"]).prod() - 1)
        rf = float((1 + window["RF"]).prod() - 1)
        rows.append({"date": date, "Mkt-RF": mkt, "SMB": smb, "HML": hml, "RF": rf})
    ff_weekly = pd.DataFrame(rows).set_index("date")
    merged = pd.DataFrame({"ret": weekly_ret}).join(ff_weekly, how="inner")
    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                          merged["SMB"].values, merged["HML"].values])
    beta, se, tvals = ols_alpha_beta(y, X)
    alpha_annualized = (1 + beta[0]) ** 52 - 1
    print(f"\n--- {label} FF3回归 (n={len(merged)}周) ---")
    for name, b, t in zip(["alpha", "Mkt-RF", "SMB", "HML"], beta, tvals):
        print(f"  {name}: 系数={b:.6f}, t值={t:.2f}")
    print(f"  年化alpha: {alpha_annualized*100:.2f}%, t值={tvals[0]:.2f} "
          f"({'显著' if abs(tvals[0]) > 2 else '不显著'})")


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    spy_close = cache.get("spy")

    with open(PIT_CACHE, "rb") as f:
        pit_cache = pickle.load(f)
    sector_map = pit_cache["sector_map"]

    n_workers = max(1, os.cpu_count() or 1)
    print("跑动量策略(2009-2014独立区间，约30分钟)...")
    mom_results = run_walk_forward_backtest(
        c_df, o_df, h_df, l_df, v_df, vix,
        enable_shap=False, n_workers=n_workers, xgb_n_jobs=1,
        enable_earnings_blackout=False, sector_map=sector_map,
    )
    with open(MOM_OUT, "wb") as f:
        pickle.dump(mom_results, f)
    mom_df = pd.DataFrame(mom_results)[["date", "weekly_return"]]
    mom_df["date"] = pd.to_datetime(mom_df["date"])
    mom_df = mom_df.set_index("date")
    print(f"动量策略: {len(mom_df)}周")

    mom_rets = mom_df["weekly_return"].values
    obs_m, ci_m, p0_m = block_bootstrap_sharpe(mom_rets)
    print(f"[纯动量 2009-2014] Sharpe={obs_m:.3f}, 95%CI=[{ci_m[0]:.3f},{ci_m[1]:.3f}], P(<=0)={p0_m:.1%}")

    print("\n跑价值策略(2009-2014独立区间)...")
    all_tickers = list(c_df.columns)
    fake_membership = {c_df.index[i].strftime("%Y-%m-%d"): all_tickers for i in range(len(c_df))}
    val_results, daily_nav = backtest_value_strategy(c_df, fake_membership, spy_close, rebalance_days=21, top_n=30)
    val_rets = [r["period_ret"] for r in val_results]
    obs_v, ci_v, p0_v = block_bootstrap_sharpe(val_rets, periods_per_year=252 / 21)
    print(f"[纯价值(月度) 2009-2014] n={len(val_rets)}, Sharpe={obs_v:.3f}, "
          f"95%CI=[{ci_v[0]:.3f},{ci_v[1]:.3f}], P(<=0)={p0_v:.1%}")

    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
    week_pairs = get_weekly_trading_dates(c_df.index)
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
    print(f"\n对齐后共 {len(merged)} 周，动量vs价值相关系数: {merged['weekly_return'].corr(merged['value_ret']):.4f}")

    combo = BEST_WEIGHT * merged["value_ret"] + (1 - BEST_WEIGHT) * merged["weekly_return"]
    obs_c, ci_c, p0_c = block_bootstrap_sharpe(combo.values)
    print(f"\n[组合(value_weight=0.5) 2009-2014] n={len(combo)}周, Sharpe={obs_c:.3f}, "
          f"95%CI=[{ci_c[0]:.3f},{ci_c[1]:.3f}], P(<=0)={p0_c:.1%}")

    print("\n拉取Fama-French三因子数据做回归...")
    ff = fetch_ff3_daily()
    regress_ff3(combo, ff, "组合策略 2009-2014")


if __name__ == "__main__":
    main()
