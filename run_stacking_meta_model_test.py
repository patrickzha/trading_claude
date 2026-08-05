"""
Cycle42大修大补②(框架改动，新组合机制——stacking元模型)：用逻辑回归把
已测试过的多个独立信号(价值复合排名、低波动排名、52周高点邻近度)融合
成一个概率化综合评分，代替第170条的简单z-score加权求和。这是"元模型"
(meta-model)这个不同的组合方式——用历史数据(trailing窗口)训练每个
信号的实际预测能力再加权，而不是手工指定等权，理论上如果某个信号
历史上更可靠，元模型应该自动给它更高的权重。

标签构造：下一个调仓期该股票收益率是否跑赢同期候选池中位数(二分类)，
训练用trailing窗口(不用未来数据)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from us_stock_screener import _compute_fundamental_features
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
REBALANCE_DAYS = 63
TOP_N = 30
LOOKBACK_52W = 252
VOL_WINDOW = 60
TRAIN_MIN_SAMPLES = 200


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def compute_features(price_df, valid_tickers, rebalance_date):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna()
        if len(px) < LOOKBACK_52W + 5:
            continue
        curr_price = px.iloc[-1]
        high_52w = px.tail(LOOKBACK_52W).max()
        proximity = curr_price / high_52w if high_52w > 0 else np.nan

        vol_px = px.tail(VOL_WINDOW)
        vol = float(vol_px.pct_change().dropna().std()) if len(vol_px) >= VOL_WINDOW - 5 else np.nan

        feat = _compute_fundamental_features(t, rebalance_date, curr_price)
        pe, pb, roe = feat["pe_ratio"], feat["pb_ratio"], feat["roe"]
        has_value = (pe is not None and not np.isnan(pe) and 0 < pe < 60
                     and pb is not None and not np.isnan(pb) and pb > 0
                     and roe is not None and not np.isnan(roe))
        rows.append({"ticker": t, "proximity": proximity, "vol": vol,
                     "pe": pe if has_value else np.nan, "pb": pb if has_value else np.nan,
                     "roe": roe if has_value else np.nan})
    return pd.DataFrame(rows)


def zscore(s):
    std = s.std()
    return (s - s.mean()) / std if std and not np.isnan(std) else pd.Series(0.0, index=s.index)


def prep_feature_matrix(df):
    df = df.copy()
    df["z_proximity"] = zscore(df["proximity"])
    df["z_vol"] = -zscore(df["vol"])
    has_val = df["pe"].notna()
    df["z_value"] = 0.0
    if has_val.sum() >= 10:
        rank_pe = df.loc[has_val, "pe"].rank(ascending=True)
        rank_pb = df.loc[has_val, "pb"].rank(ascending=True)
        rank_roe = df.loc[has_val, "roe"].rank(ascending=False)
        composite_rank = (rank_pe + rank_pb + rank_roe) / 3
        df.loc[has_val, "z_value"] = zscore(-composite_rank)
    return df[["ticker", "z_proximity", "z_vol", "z_value"]].fillna(0.0)


def backtest_stacking(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = LOOKBACK_52W + 5
    running_nav = 1.0

    train_X, train_y = [], []
    model = None

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        feat_df = compute_features(price_df, valid, rebalance_date)
        if len(feat_df) < TOP_N:
            rebalance_idx += rebalance_days
            continue
        X_df = prep_feature_matrix(feat_df)

        if model is None and len(train_X) >= TRAIN_MIN_SAMPLES:
            model = LogisticRegression(max_iter=200)
            model.fit(np.array(train_X), np.array(train_y))

        if model is not None:
            probs = model.predict_proba(X_df[["z_proximity", "z_vol", "z_value"]].values)[:, 1]
            X_df = X_df.assign(score=probs)
        else:
            X_df = X_df.assign(score=X_df["z_proximity"] + X_df["z_vol"] + X_df["z_value"])

        picks = X_df.sort_values("score", ascending=False)["ticker"].head(TOP_N).tolist()
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

        period_days = index[rebalance_idx:min(rebalance_idx + rebalance_days, n)]
        entry_prices = {t: price_df[t].loc[:rebalance_date].dropna().iloc[-1] for t in picks}
        fwd_prices = {}
        for t in picks:
            future = price_df[t].loc[rebalance_date:].dropna()
            if len(future) > 1:
                fwd_prices[t] = future.iloc[min(rebalance_days, len(future) - 1)]

        median_ret = np.median([fwd_prices[t] / entry_prices[t] - 1 for t in picks if t in fwd_prices])
        for t in picks:
            if t not in fwd_prices:
                continue
            fwd_ret = fwd_prices[t] / entry_prices[t] - 1
            label = 1 if fwd_ret > median_ret else 0
            row = X_df[X_df["ticker"] == t]
            if len(row):
                train_X.append(row[["z_proximity", "z_vol", "z_value"]].values[0])
                train_y.append(label)

        for d in period_days:
            day_val, total_w = 0.0, 0.0
            for t, p0 in entry_prices.items():
                if d in price_df.index and t in price_df.columns:
                    p = price_df.at[d, t]
                    if pd.notna(p) and p0:
                        day_val += p / p0
                        total_w += 1
            if total_w > 0:
                daily_nav.append((d, running_nav * (day_val / total_w)))
        if daily_nav:
            running_nav = daily_nav[-1][1]
        rebalance_idx += rebalance_days

    return daily_nav


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def metrics(nav_list):
    nav = np.array([n for _, n in nav_list])
    n_years = len(nav) / 252
    cagr = float((nav[-1] / nav[0]) ** (1 / n_years) - 1) if n_years > 0 else None
    rets = np.diff(nav) / nav[:-1]
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252))
    mdd = true_mdd(nav)
    return cagr, sharpe, mdd


def main():
    with open("/Users/zhang/Desktop/trading/sp500_membership_added_dates.json") as f:
        import json
        added_dates_raw = json.load(f)
    added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

    periods = [
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]

    for label, price_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        nav = backtest_stacking(price_df, membership)
        print(f"\n=== {label} ===")
        if len(nav) < 30:
            print("  样本不足，跳过")
            continue
        cagr, sharpe, mdd = metrics(nav)
        print(f"  Stacking元模型组合: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")


if __name__ == "__main__":
    main()
