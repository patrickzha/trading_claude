"""
Cycle33大修大补①(框架改动，新组合构建架构)：截面多因子复合评分组合——
用动量(6个月价格收益)、价值(PE/PB/ROE复合排名)、低波动(60日已实现波动率)
三个信号，在每个调仓日对候选池做横截面z-score标准化后加权求和，选
复合分最高的N只票、等权持有，构建**单一统一组合**。

这是对第60条"三元组合是价值单独→加动量→加低波动的三次迭代式改进，
不是三个独立候选，天然更容易过拟合"这个方法论顾虑的一次真正架构级别
回应——现有combo_strategy.py是"三条独立backtest各自选股、再按NAV等权
混合"，本质上是三个子策略的资金分配问题；这里改成"用三个信号在个股
层面直接构造一个复合评分、一次性选股"，是完全不同的组合构建方式，
如果两种架构能独立得到类似的正面结论，会大幅加强现有推荐配置的可信度；
如果本质上表现类似或更差，也是有价值的负面/中性结果，会削弱"三元组合
不是过拟合"这个论证。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from us_stock_screener import _compute_fundamental_features
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
REBALANCE_DAYS = 63
TOP_N = 30
VOL_WINDOW = 60
MOM_WINDOW = 126


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def compute_composite_scores(price_df, valid_tickers, rebalance_date):
    rows = []
    for t in valid_tickers:
        if t not in price_df.columns:
            continue
        px = price_df[t].loc[:rebalance_date].dropna()
        if len(px) < MOM_WINDOW + 5:
            continue
        curr_price = px.iloc[-1]
        mom = curr_price / px.iloc[-MOM_WINDOW] - 1
        vol_px = px.tail(VOL_WINDOW)
        if len(vol_px) < VOL_WINDOW - 5:
            continue
        vol = float(vol_px.pct_change().dropna().std())

        feat = _compute_fundamental_features(t, rebalance_date, curr_price)
        pe, pb, roe = feat["pe_ratio"], feat["pb_ratio"], feat["roe"]
        has_value = (pe is not None and not np.isnan(pe) and 0 < pe < 60
                     and pb is not None and not np.isnan(pb) and pb > 0
                     and roe is not None and not np.isnan(roe))
        rows.append({"ticker": t, "mom": mom, "vol": vol,
                     "pe": pe if has_value else np.nan, "pb": pb if has_value else np.nan,
                     "roe": roe if has_value else np.nan})

    if len(rows) < TOP_N:
        return [r["ticker"] for r in rows]

    df = pd.DataFrame(rows)
    df["z_mom"] = zscore(df["mom"])
    df["z_vol"] = -zscore(df["vol"])

    has_val = df["pe"].notna()
    df["z_value"] = 0.0
    if has_val.sum() >= 10:
        val_sub = df[has_val]
        rank_pe = val_sub["pe"].rank(ascending=True)
        rank_pb = val_sub["pb"].rank(ascending=True)
        rank_roe = val_sub["roe"].rank(ascending=False)
        composite_rank = (rank_pe + rank_pb + rank_roe) / 3
        df.loc[has_val, "z_value"] = zscore(-composite_rank)

    df["composite"] = df["z_mom"] + df["z_value"] + df["z_vol"]
    df = df.sort_values("composite", ascending=False)
    return df["ticker"].head(TOP_N).tolist()


def backtest_composite(price_df, membership, rebalance_days=REBALANCE_DAYS):
    index = price_df.index
    n = len(index)
    membership_keys = sorted(membership.keys())
    daily_nav = []
    rebalance_idx = MOM_WINDOW + 5
    running_nav = 1.0

    while rebalance_idx + rebalance_days < n:
        rebalance_date = index[rebalance_idx]
        date_str = rebalance_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            rebalance_idx += rebalance_days
            continue

        picks = compute_composite_scores(price_df, valid, rebalance_date)
        picks = [p for p in picks if p in price_df.columns]
        if len(picks) < 5:
            rebalance_idx += rebalance_days
            continue

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
        ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
        ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
        ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
    ]

    for label, price_cache_path, is_pit in periods:
        with open(price_cache_path, "rb") as f:
            cache = pickle.load(f)
        price_df, spy_close = cache["c"], cache.get("spy")
        if is_pit:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                            cache["added_dates"], cache["removal_dates"])
        else:
            membership = compute_point_in_time_membership(price_df.index, list(price_df.columns), added_dates, {})

        print(f"\n=== {label}：跑截面复合评分组合(季度调仓) ===")
        nav = backtest_composite(price_df, membership)
        if len(nav) < 30:
            print(f"  样本不足(n={len(nav)})，跳过")
            continue
        cagr, sharpe, mdd = metrics(nav)
        spy_aligned = spy_close.reindex([d for d, _ in nav]).ffill().dropna() if spy_close is not None else None
        spy_cagr = None
        if spy_aligned is not None and len(spy_aligned) > 1:
            n_years = len(nav) / 252
            spy_cagr = float((spy_aligned.iloc[-1] / spy_aligned.iloc[0]) ** (1 / n_years) - 1)
        print(f"  复合评分组合: CAGR={cagr*100:.2f}%, Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%, n_days={len(nav)}")
        if spy_cagr is not None:
            print(f"  同期SPY CAGR={spy_cagr*100:.2f}% (超额={ (cagr-spy_cagr)*100:+.2f}%)")


if __name__ == "__main__":
    main()
