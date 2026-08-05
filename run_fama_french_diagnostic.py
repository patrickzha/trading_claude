"""
Fama-French三因子分解诊断：这次会话新发现的"长周期价值投资"策略（见README
"验证结论"第38条）CAGR明显跑赢SPY，但用的选股逻辑(低PE/低PB/高ROE)本身
就是经典的HML(高账面市值比减低账面市值比)价值因子的直接实现——这个诊断
要回答的问题是：这个策略的超额收益，有多少只是"暴露了一个学术界几十年前
就发表过的已知因子"(不算新发现，只是换个壳子实现HML)，有多少是HML因子
暴露之外真正的alpha。

用Fama-French三因子模型(Mkt-RF, SMB, HML)对策略的日收益率做OLS回归，
alpha(截距项，年化)代表因子暴露之外解释不了的部分；因子载荷(beta)代表
这个策略在多大程度上就是一个换皮的因子基金。
"""
from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pandas as pd
import requests

from value_investing import backtest_value_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def fetch_ff3_daily():
    r = requests.get(FF_URL, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(z.namelist()[0]) as f:
        content = f.read().decode("latin1")
    lines = content.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "")
    data_lines = lines[start + 1:end]
    rows = []
    for l in data_lines:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) != 5 or not parts[0].isdigit():
            continue
        date = pd.Timestamp(parts[0])
        rows.append([date] + [float(x) / 100.0 for x in parts[1:]])
    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"]).set_index("date")
    return df


def ols_alpha_beta(y: np.ndarray, X: np.ndarray):
    """普通最小二乘，X已含截距列。返回系数、标准误、t值。"""
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    return beta, se, tvals


def main():
    print("拉取Fama-French三因子日数据...")
    ff = fetch_ff3_daily()
    print(f"FF3数据: {ff.index[0].date()} ~ {ff.index[-1].date()}, {len(ff)}天")

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    spy_close = cache.get("spy")
    added_dates = cache["added_dates"]
    removal_dates = cache["removal_dates"]
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, removal_dates)

    print("跑价值策略(top30_semiannual)拿逐日净值...")
    results, daily_nav = backtest_value_strategy(price_df, membership, spy_close,
                                                   rebalance_days=126, top_n=30)
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"]).set_index("date")
    nav_df["ret"] = nav_df["nav"].pct_change()
    nav_df = nav_df.dropna()

    merged = nav_df.join(ff, how="inner")
    print(f"对齐后共 {len(merged)} 个交易日")
    if len(merged) < 30:
        print("对齐后样本太少，无法做有意义的回归")
        return

    y = (merged["ret"] - merged["RF"]).values
    X = np.column_stack([np.ones(len(merged)), merged["Mkt-RF"].values,
                          merged["SMB"].values, merged["HML"].values])
    beta, se, tvals = ols_alpha_beta(y, X)
    names = ["alpha(日)", "beta_Mkt-RF", "beta_SMB", "beta_HML"]

    print("\n" + "=" * 60)
    print(" Fama-French三因子回归结果(价值策略超额收益)")
    print("=" * 60)
    for name, b, s, t in zip(names, beta, se, tvals):
        print(f"  {name}: 系数={b:.6f}, 标准误={s:.6f}, t值={t:.2f}")
    alpha_annualized = (1 + beta[0]) ** 252 - 1
    print(f"\n  年化alpha: {alpha_annualized*100:.2f}%")
    print(f"  alpha的t值: {tvals[0]:.2f} ({'显著(|t|>2)' if abs(tvals[0]) > 2 else '不显著(|t|<=2)，样本量小/噪声大，不能确认真实存在'})")
    print(f"  HML载荷t值: {tvals[3]:.2f} ({'显著' if abs(tvals[3]) > 2 else '不显著'})")


if __name__ == "__main__":
    main()
