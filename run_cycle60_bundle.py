"""
Cycle60大修大补①+小修(a)(b)(c)(d)合并脚本。

大修①：新推荐配置(value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05,
value_weighting="hrp", value_rebalance_days=5)是否跟旧配置(5%/5%)一样，
对成本假设/候选池规模/调仓频率不敏感——第252/266/269/270条的稳健性验证
全部是在旧配置上做的，不能假设新配置自动继承这些结论，需要重新走一遍。
这次顺手给`combo_strategy.run_combo_strategy`加了`cost_bi`透传参数(之前
没有暴露，成本硬编码在`backtest_value_strategy`/`backtest_low_vol_strategy`
内部)，默认值0.003，已做回归测试确认不改变现有默认行为。

小修(a)：value_stop_loss_pct精细网格(2%/2.5%/3%/3.5%/4%)，确认第276条
选中的3%是不是局部最优，而不是随便测了一个点。

小修(b)：新配置(3%/5%)相对旧配置(5%/5%)的Fama-French三因子alpha分解，
检验这次的改善是新的风险调整后收益来源，还是纯粹的风险管理效应(跟这次
会话反复出现的"组合优化类改进提升Sharpe靠风险管理不是新收益"这个主题
一致)。

小修(c)：value_stop_loss_pct=0.03在min_variance/risk_parity两种权重方案
下是否同样成立，不是HRP专属的现象——第276条只在HRP权重下测过。
"""
from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests

from combo_strategy import run_combo_strategy
from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from stats_utils import to_weekly

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership


NEW_BASE = dict(value_weighting="hrp", value_rebalance_days=5,
                value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

print(f"\n{'='*70}\n 大修①-1: 成本敏感性(新配置3%/5%, combo_strategy层面)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for cb in [0.003, 0.005, 0.01, 0.02]:
        r = run_combo_strategy(price_df, membership, spy_close, mom_results, cost_bi=cb, **NEW_BASE)
        print(f"  cost_bi={cb}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 大修①-2: 候选池规模敏感性(新配置3%/5%)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for tn in [20, 30, 40]:
        r = run_combo_strategy(price_df, membership, spy_close, mom_results,
                                value_top_n=tn, low_vol_top_n=tn, **NEW_BASE)
        print(f"  top_n={tn}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 大修①-3: 调仓频率敏感性(新配置3%/5%)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    base_no_rebal = {k: v for k, v in NEW_BASE.items() if k != "value_rebalance_days"}
    for rd in [5, 10, 15, 21]:
        r = run_combo_strategy(price_df, membership, spy_close, mom_results,
                                value_rebalance_days=rd, **base_no_rebal)
        print(f"  value_rebalance_days={rd}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 小修(a): value_stop_loss_pct精细网格(2%-4%)\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    base_no_sl = {k: v for k, v in NEW_BASE.items() if k != "value_stop_loss_pct"}
    for sl in [0.02, 0.025, 0.03, 0.035, 0.04]:
        r = run_full_system(price_df, membership, spy_close, mom_results,
                             value_stop_loss_pct=sl, **base_no_sl)
        print(f"  value_stop_loss_pct={sl}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print(f"\n{'='*70}\n 小修(b): 新配置(3%/5%) vs 旧配置(5%/5%) Fama-French三因子alpha分解\n{'='*70}")


def fetch_ff3_daily():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    r = requests.get(url, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(z.namelist()[0]) as f:
        content = f.read().decode("latin1")
    lines = content.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "")
    rows = []
    for l in lines[start + 1:end]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        rows.append([pd.Timestamp(parts[0])] + [float(x) / 100.0 for x in parts[1:]])
    return pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"]).set_index("date")


def ols_alpha_beta(y, X):
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, beta / se


try:
    ff = fetch_ff3_daily()
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
        price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
        print(f"\n--- {label} ---")
        r_old = run_full_system(price_df, membership, spy_close, mom_results,
                                 value_weighting="hrp", value_rebalance_days=5,
                                 value_stop_loss_pct=0.05, low_vol_stop_loss_pct=0.05)
        r_new = run_full_system(price_df, membership, spy_close, mom_results, **NEW_BASE)
        for cname, r in [("旧5%/5%", r_old), ("新3%/5%", r_new)]:
            nav = r["full_nav"]
            rets = nav.pct_change().dropna()
            merged = pd.DataFrame({"ret": rets}).join(ff, how="inner")
            y = (merged["ret"] - merged["RF"]).to_numpy()
            X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"], merged["SMB"], merged["HML"]])
            beta, tvals = ols_alpha_beta(y, X)
            ann_alpha = beta[0] * 252
            print(f"  {cname}: 年化alpha={ann_alpha*100:.2f}%, t={tvals[0]:.3f}, "
                  f"beta_Mkt={beta[1]:.3f}, beta_SMB={beta[2]:.3f}, beta_HML={beta[3]:.3f}, n={len(merged)}")
except Exception as e:
    print(f"  [警告] FF3数据拉取或回归失败: {type(e).__name__}: {e}")

print(f"\n{'='*70}\n 小修(c): value_stop_loss_pct=0.03在min_variance/risk_parity权重下是否同样成立\n{'='*70}")
for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
    price_df, spy_close, mom_results, membership = load_period(price_cache_path, mom_cache_path, is_pit)
    print(f"\n--- {label} ---")
    for weighting in ["min_variance", "risk_parity", "hrp"]:
        for sl in [0.05, 0.03]:
            r = run_full_system(price_df, membership, spy_close, mom_results,
                                 value_weighting=weighting, value_rebalance_days=5,
                                 value_stop_loss_pct=sl, low_vol_stop_loss_pct=0.05)
            print(f"  {weighting}, stop={sl}: Sharpe={r['Sharpe']:.3f}, MaxDD={r['MaxDD']*100:.2f}%")

print("\n全部完成。")
