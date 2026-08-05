"""
Cycle22小修小补：用PIT修正后的动量数据，重新做三元组合的block bootstrap
Sharpe显著性检验(跟第111条的Fama-French检验是两种不同的显著性度量，
互相补充——bootstrap测的是"这个策略本身赚不赚钱"，Fama-French测的是
"赚的这部分钱有多少是已知因子解释不了的")。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from combo_strategy import run_combo_strategy
from backtest import get_weekly_trading_dates

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def block_bootstrap_sharpe(rets, block=4, n_boot=3000, periods_per_year=52.0, seed=42):
    rng = np.random.default_rng(seed)
    rets = np.array(rets)
    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([rets[i:i + block] for i in idx])[:n]
        boot.append(sample.mean() / (sample.std() + 1e-9) * np.sqrt(periods_per_year))
    boot = np.array(boot)
    obs = rets.mean() / (rets.std() + 1e-9) * np.sqrt(periods_per_year)
    return obs, float(np.mean(boot <= 0))


def to_weekly(nav_series, price_df):
    week_pairs = get_weekly_trading_dates(price_df.index)
    rows = []
    for monday, friday in week_pairs:
        try:
            a, b = nav_series.loc[:monday].iloc[-1], nav_series.loc[:friday].iloc[-1]
        except IndexError:
            continue
        rows.append({"date": monday, "ret": b / a - 1})
    return pd.DataFrame(rows).set_index("date")["ret"]


periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache_PRE_PIT_BACKUP.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache_PRE_PIT_BACKUP.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]

for label, price_cache_path, old_mom_path, new_mom_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    print(f"\n=== {label} ===")
    for tag, mom_path in [("修正前", old_mom_path), ("修正后", new_mom_path)]:
        with open(mom_path, "rb") as f:
            mom = pickle.load(f)
        report = run_combo_strategy(price_df, membership, spy_close, mom)
        weekly_ret = to_weekly(report["combo_nav"], price_df).dropna()
        obs, p0 = block_bootstrap_sharpe(weekly_ret.values)
        print(f"  {tag}: bootstrap Sharpe={obs:.3f}, P(<=0)={p0:.1%} ({'显著' if p0 < 0.05 else '不显著'})")
