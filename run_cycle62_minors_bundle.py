"""
Cycle62小修(a)(b)(c)合并：EV/EBIT因子(第284条)的后续标准化检验。

小修(a)：EV/EBIT + 现有P/E-P/B-ROE价值因子二元组合(各半资金独立选股，
跟第203条偏度+价值二元组合同样的方法论)，检验两个低相关(第284条重叠度
23.3%)的价值构造混合后能不能获得真正的分散化收益。

小修(b)：EV/EBIT因子的Fama-French三因子alpha分解，检验是不是纯粹的HML
暴露(跟第39条原始价值因子当年的诊断问题一样)还是有独立于HML之外的alpha。

小修(c)：EV/EBIT因子的成本敏感性快速检查(0.3%/0.5%/1.0%三档)。
"""
from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests

from ev_ebit_investing import select_ev_ebit_portfolio
from value_investing import select_value_portfolio
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

REBALANCE_DAYS = 126
TOP_N = 30


def backtest_simple(price_df, membership, select_fn, cost_bi=0.0, **select_kwargs):
    index = price_df.index
    n = len(index)
    rebalance_idx = 252
    daily_nav = []
    running_nav = 1.0
    prev_picks = set()
    while rebalance_idx + REBALANCE_DAYS < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_fn(price_df, valid_tickers, rebalance_date, top_n=TOP_N, **select_kwargs)
        picks = [p for p in picks if p in price_df.columns]
        if cost_bi > 0 and picks:
            turnover = len(set(picks) - prev_picks) / len(picks) if prev_picks else 1.0
            running_nav *= (1 - turnover * cost_bi)
        prev_picks = set(picks)

        period_days = index[rebalance_idx:min(rebalance_idx + REBALANCE_DAYS, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks
                         if len(price_df[t].loc[:rebalance_date].dropna()) > 0}
        for d in period_days:
            vals = []
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        vals.append(p / p0)
            if vals:
                daily_nav.append((d, running_nav * np.mean(vals)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += REBALANCE_DAYS

    nav_series = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")["nav"]
    return nav_series


def metrics(nav_series):
    rets = nav_series.pct_change().dropna()
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    n_years = len(nav_series) / 252
    cagr = float(nav_series.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    peak = nav_series.cummax()
    mdd = float(((nav_series - peak) / peak).min())
    return sharpe, cagr, mdd


with open(CACHE_PATH, "rb") as f:
    cache = pickle.load(f)
price_df, spy_close = cache["c"], cache.get("spy")
membership = compute_point_in_time_membership(
    price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])

print(f"{'='*70}\n 小修(a): EV/EBIT + 现有价值因子二元组合\n{'='*70}")
nav_ev = backtest_simple(price_df, membership, select_ev_ebit_portfolio)
nav_val = backtest_simple(price_df, membership, select_value_portfolio)
common = nav_ev.index.intersection(nav_val.index)
combo_nav = ((nav_ev.loc[common] / nav_ev.loc[common].iloc[0]) + (nav_val.loc[common] / nav_val.loc[common].iloc[0])) / 2
s_ev, c_ev, m_ev = metrics(nav_ev)
s_val, c_val, m_val = metrics(nav_val)
s_combo, c_combo, m_combo = metrics(combo_nav)
print(f"  EV/EBIT单独: Sharpe={s_ev:.3f}, CAGR={c_ev*100:.2f}%, MaxDD={m_ev*100:.2f}%")
print(f"  价值因子单独: Sharpe={s_val:.3f}, CAGR={c_val*100:.2f}%, MaxDD={m_val*100:.2f}%")
print(f"  二元组合(各半): Sharpe={s_combo:.3f}, CAGR={c_combo*100:.2f}%, MaxDD={m_combo*100:.2f}%")

print(f"\n{'='*70}\n 小修(b): EV/EBIT因子Fama-French三因子alpha分解\n{'='*70}")
try:
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
    ff = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"]).set_index("date")

    rets = nav_ev.pct_change().dropna()
    merged = pd.DataFrame({"ret": rets}).join(ff, how="inner")
    y = (merged["ret"] - merged["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"], merged["SMB"], merged["HML"]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    ann_alpha = beta[0] * 252
    print(f"  EV/EBIT因子: 年化alpha={ann_alpha*100:.2f}%, t={tvals[0]:.3f}, "
          f"beta_Mkt={beta[1]:.3f}, beta_SMB={beta[2]:.3f}, beta_HML={beta[3]:.3f}, n={len(merged)}")
except Exception as e:
    print(f"  [警告] FF3检验失败: {type(e).__name__}: {e}")

print(f"\n{'='*70}\n 小修(c): EV/EBIT因子成本敏感性\n{'='*70}")
for cb in [0.003, 0.005, 0.01]:
    nav = backtest_simple(price_df, membership, select_ev_ebit_portfolio, cost_bi=cb)
    s, c, m = metrics(nav)
    print(f"  cost_bi={cb}: Sharpe={s:.3f}, CAGR={c*100:.2f}%, MaxDD={m*100:.2f}%")

print("\n全部完成。")
