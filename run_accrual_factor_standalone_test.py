"""
Cycle69大修大补②(框架改动，隔离测试——单独的应计利润异象)：第134条测的是
"毛利率+经营利润率+应计比率+资产增长率"四合一综合质量分，结果是负面的，
但从未单独隔离测试过"应计利润比率"这一个维度本身——Sloan(1996)应计利润
异象是四个维度里学术证据最独立、最经典的一个(利润"含金量"越低，即净利润
里现金流占比越少，未来收益越差)，这次单独测，不跟另外三个维度混在一起，
避免像第134条那样被其他维度的噪声稀释掉可能存在的信号。用IC/分组前瞻
收益率标准化检验(先用最直接的方式测有没有信号，不直接上完整回测)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quality_investing import _compute_quality_features
from run_survivorship_bias_fix_validation import compute_point_in_time_membership, CACHE_PATH

HORIZON_DAYS = 126  # 半年度调仓周期，跟第134条同等持有期
STEP = 21


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    membership = compute_point_in_time_membership(
        price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    membership_keys = sorted(membership.keys())

    index = price_df.index
    n = len(index)
    all_ic = []
    all_pairs = []

    for i in range(260, n - HORIZON_DAYS, STEP):
        decision_date = index[i]
        date_str = decision_date.strftime("%Y-%m-%d")
        valid = membership.get(date_str)
        if valid is None:
            candidates = [k for k in membership_keys if k <= date_str]
            if not candidates:
                continue
            valid = membership[max(candidates)]
        valid = [t for t in valid if t in price_df.columns]
        if len(valid) < 30:
            continue

        rows = []
        for t in valid:
            feat = _compute_quality_features(t, decision_date)
            ar = feat["accrual_ratio"]
            if np.isnan(ar):
                continue
            px = price_df[t].loc[:decision_date].dropna()
            if len(px) == 0:
                continue
            try:
                fwd = price_df[t].iloc[i + HORIZON_DAYS] / px.iloc[-1] - 1
            except Exception:
                continue
            if pd.isna(fwd):
                continue
            rows.append({"ticker": t, "accrual_ratio": ar, "fwd_ret": fwd})

        if len(rows) < 30:
            continue
        df = pd.DataFrame(rows)
        ic, _ = spearmanr(df["accrual_ratio"], df["fwd_ret"])
        if not np.isnan(ic):
            all_ic.append(ic)
        all_pairs.append(df)

    ic_arr = np.array(all_ic)
    print(f"应计利润比率(越低越好) IC均值={ic_arr.mean():.4f}, 标准差={ic_arr.std():.4f}, "
          f"IC<0占比(方向正确，应计低预示未来收益高)={np.mean(ic_arr < 0)*100:.1f}%, n={len(ic_arr)}个决策点")

    from scipy.stats import ttest_1samp
    t_stat, p_val = ttest_1samp(ic_arr, 0)
    print(f"t检验: t={t_stat:.3f}, p={p_val:.4f}")

    # 五分位分组前瞻收益(汇总所有决策点的候选，按应计比率分5档)
    all_df = pd.concat(all_pairs, ignore_index=True)
    all_df["quintile"] = pd.qcut(all_df["accrual_ratio"], 5, labels=False, duplicates="drop")
    print("\n应计利润比率五分位分组平均前瞻收益(Q0=应计最低/质量最高 ... Q4=应计最高/质量最低):")
    print(all_df.groupby("quintile")["fwd_ret"].mean() * 100)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
