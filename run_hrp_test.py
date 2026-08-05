"""
Cycle31大修大补①(框架改动)：层次风险平价(Hierarchical Risk Parity, HRP,
López de Prado 2016)——第155/156条的风险平价(用scipy.optimize数值求解
等风险贡献)在候选池风险异质化程度不同的场景下表现不完全稳定，HRP是专门
为了解决"传统均值方差/风险平价类方法对协方差矩阵估计误差敏感"这个问题
设计的替代框架：不需要对协方差矩阵求逆(数值上更稳定)，用层次聚类
先把资产按相关性分组，再用递归二分配权重(组内、组间都按方差反比分配)，
理论上应该比第155条的数值优化版风险平价更抗协方差估计噪声。这是这次
会话第三个真正意义上的组合构建框架(继最小方差、风险平价之后)。
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from value_investing import select_value_portfolio
from stats_utils import true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
COV_WINDOW = 60


def _get_quasi_diag(link):
    """从linkage矩阵里取出叶子节点的seriation顺序(把相关性高的资产排在
    相邻位置)，标准HRP第二步。"""
    link = link.astype(int)
    n = link.shape[0] + 1
    sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_idx.max() >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        df0 = sort_idx[sort_idx >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_idx[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_idx = pd.concat([sort_idx, df1])
        sort_idx = sort_idx.sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.tolist()


def _get_cluster_var(cov, items):
    sub_cov = cov.loc[items, items]
    ivp = 1.0 / np.diag(sub_cov)
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _recursive_bisection(cov, sort_ix_tickers):
    """标准HRP递归二分：只有长度>1的簇才继续切分，全部退化成单元素簇后
    停止——第一版忘了这个终止条件，单元素簇对半切(0,0)/(0,1)会重新产出
    自己，导致死循环，这里是修复后的版本。"""
    weights = pd.Series(1.0, index=sort_ix_tickers)
    clusters = [sort_ix_tickers]
    while any(len(c) > 1 for c in clusters):
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                new_clusters.append(c)
                continue
            mid = len(c) // 2
            c0, c1 = c[:mid], c[mid:]
            var0 = _get_cluster_var(cov, c0)
            var1 = _get_cluster_var(cov, c1)
            alpha = 1 - var0 / (var0 + var1)
            weights[c0] *= alpha
            weights[c1] *= 1 - alpha
            new_clusters.append(c0)
            new_clusters.append(c1)
        clusters = new_clusters
    return weights


def hrp_weights(returns_window: pd.DataFrame) -> pd.Series:
    """完整HRP流程：相关性->距离矩阵->层次聚类->quasi-diagonalization->
    递归二分配权重。任何一步失败(比如聚类退化)都fail-safe退化成等权。"""
    try:
        corr = returns_window.corr()
        cov = returns_window.cov()
        dist = np.sqrt(0.5 * (1 - corr))
        condensed = squareform(dist.values, checks=False)
        link = linkage(condensed, method="single")
        sort_ix = _get_quasi_diag(link)
        sort_ix_tickers = [returns_window.columns[i] for i in sort_ix]
        w = _recursive_bisection(cov, sort_ix_tickers)
        w = w.reindex(returns_window.columns)
        if w.isna().any() or w.sum() <= 0:
            return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)
        return w / w.sum()
    except Exception:
        return pd.Series(1.0 / len(returns_window.columns), index=returns_window.columns)


def backtest_weighted_value(price_df, membership, spy_close, rebalance_days=126, top_n=30,
                             cost_bi=0.003, weighting="equal"):
    index = price_df.index
    n = len(index)
    daily_nav = []
    rebalance_idx = 252 + COV_WINDOW
    running_nav = 1.0
    prev_weights = None

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid_tickers = membership.get(date_str)
        if valid_tickers is None:
            keys = sorted(membership.keys())
            valid_tickers = membership[max([k for k in keys if k <= date_str], default=keys[0])]

        picks = select_value_portfolio(price_df, valid_tickers, rebalance_date, top_n=top_n)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        if weighting == "hrp":
            hist_window = price_df[picks].loc[:rebalance_date].tail(COV_WINDOW + 1)
            rets_window = hist_window.pct_change().dropna()
            valid_cols = rets_window.columns[rets_window.notna().all()]
            if len(valid_cols) >= 5:
                hw = hrp_weights(rets_window[valid_cols])
                weights = pd.Series(0.0, index=picks)
                for t in valid_cols:
                    weights[t] = hw[t]
                weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)
            else:
                weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            weights = pd.Series(1.0 / len(picks), index=picks)

        turnover = 1.0 if prev_weights is None else sum(
            abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(weights.index) | set(prev_weights.index)) / 2
        running_nav *= (1 - turnover * cost_bi)
        prev_weights = weights

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {}
        for t in picks:
            try:
                entry_prices[t] = price_df[t].loc[:rebalance_date].dropna().iloc[-1]
            except IndexError:
                continue
        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        w = weights.get(t, 0)
                        day_val += w * (p / p0)
                        total_w += w
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_max_drawdown(nav)
    return cagr, sharpe, mdd


if __name__ == "__main__":
    periods = [
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    for label, price_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        nav_equal = backtest_weighted_value(price_df, membership, spy_close, weighting="equal")
        nav_hrp = backtest_weighted_value(price_df, membership, spy_close, weighting="hrp")
        e_cagr, e_sharpe, e_mdd = metrics(nav_equal)
        h_cagr, h_sharpe, h_mdd = metrics(nav_hrp)
        print(f"\n=== {label} ===")
        print(f"  等权(现有): CAGR={e_cagr*100:.2f}%, Sharpe={e_sharpe:.3f}, MaxDD={e_mdd*100:.2f}%")
        print(f"  HRP(新框架): CAGR={h_cagr*100:.2f}%, Sharpe={h_sharpe:.3f}, MaxDD={h_mdd*100:.2f}%")
