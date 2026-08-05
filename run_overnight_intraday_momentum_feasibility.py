"""
Cycle59大修大补②(框架改动，新数据维度——收益率的日内时间分解)：把每日
收益拆成隔夜(前一日收盘→当日开盘)和日内(当日开盘→当日收盘)两段，分别
构造"隔夜动量"和"日内动量"两个信号，检验对下一周横截面收益是否有预测力。

理论依据：Lou/Polk/Skouras(2019, "A Tug of War")等文献发现，隔夜收益率
和日内收益率携带的信息不同——隔夜收益(经常伴随财报/新闻在盘后发布)倾向
于表现出动量延续，日内收益(更多是短线交易者/做市商流动性驱动)倾向于
均值回归。这次会话现有的21维技术特征全部基于收盘价构造的常规动量，
从未把"同一天内涨跌发生在什么时间段"这个维度单独拆出来看，是一个真正
不同于价格动量/基本面/宏观信号的新数据维度(不是新的比率组合，是对
已有OHLC数据做了一次此前完全没做过的时间切片)。

方法论：复用已有点位时间正确的三段区间价格缓存(含open/close)，在每个
周一决策日，用trailing 126个交易日的累计隔夜收益率/累计日内收益率分别
排序全部有效候选股，跟接下来一周(下一个周五收盘/当前收盘)的真实前瞻
收益率算Spearman相关系数(IC)，三段区间独立测，汇总统计显著性。如果
IC接近0，是一次干净的负面结果，如实记录；如果有一段或两端出现有意义
的IC，再决定要不要往下走完整策略集成测试(遵循这次会话反复用过的
"先测标准化IC、再决定要不要投入策略构建"两步流程，见第43/44/226条)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
LOOKBACK = 126
HORIZON = 5  # 一周(5个交易日)前瞻收益

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", False),
]


def _lookup_membership(membership, membership_keys, date_str):
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


all_ic_overnight = []
all_ic_intraday = []

for label, cache_path, is_pit in PERIODS:
    print(f"\n{'='*70}\n {label}\n{'='*70}")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df = cache["c"], cache["o"]
    common_cols = c_df.columns.intersection(o_df.columns)
    c_df, o_df = c_df[common_cols], o_df[common_cols]
    index = c_df.index

    if is_pit:
        membership = compute_point_in_time_membership(
            index, list(common_cols), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {index[i].strftime("%Y-%m-%d"): list(common_cols) for i in range(len(index))}
    membership_keys = sorted(membership.keys())

    # 隔夜收益: open_t / close_{t-1} - 1；日内收益: close_t / open_t - 1
    overnight_ret = (o_df / c_df.shift(1) - 1)
    intraday_ret = (c_df / o_df - 1)

    period_ic_overnight = []
    period_ic_intraday = []

    step = 5  # 每周一个决策点，跟系统周频调仓节奏一致
    for i in range(LOOKBACK + 5, len(index) - HORIZON, step):
        decision_date = index[i]
        date_str = decision_date.strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            continue
        valid = [t for t in valid if t in common_cols]
        if len(valid) < 30:
            continue

        on_window = overnight_ret[valid].iloc[i - LOOKBACK:i]
        id_window = intraday_ret[valid].iloc[i - LOOKBACK:i]
        on_score = (1 + on_window).prod() - 1
        id_score = (1 + id_window).prod() - 1

        fwd = c_df[valid].iloc[i + HORIZON] / c_df[valid].iloc[i] - 1

        valid_mask = on_score.notna() & id_score.notna() & fwd.notna()
        if valid_mask.sum() < 30:
            continue

        ic_on, _ = spearmanr(on_score[valid_mask], fwd[valid_mask])
        ic_id, _ = spearmanr(id_score[valid_mask], fwd[valid_mask])
        if not np.isnan(ic_on):
            period_ic_overnight.append(ic_on)
        if not np.isnan(ic_id):
            period_ic_intraday.append(ic_id)

    on_arr = np.array(period_ic_overnight)
    id_arr = np.array(period_ic_intraday)
    print(f"  隔夜动量IC: 均值={on_arr.mean():.4f}, 标准差={on_arr.std():.4f}, "
          f"IC>0占比={np.mean(on_arr > 0)*100:.1f}%, n={len(on_arr)}个决策点")
    print(f"  日内动量IC: 均值={id_arr.mean():.4f}, 标准差={id_arr.std():.4f}, "
          f"IC>0占比={np.mean(id_arr > 0)*100:.1f}%, n={len(id_arr)}个决策点")

    # t检验：均值IC是否显著不为0(用决策点样本自身的标准误，简单近似，
    # 决策点之间有重叠窗口不完全独立，这个t值是乐观上界，仅作参考)
    from scipy.stats import ttest_1samp
    t_on, p_on = ttest_1samp(on_arr, 0)
    t_id, p_id = ttest_1samp(id_arr, 0)
    print(f"  隔夜动量IC t检验: t={t_on:.3f}, p={p_on:.4f}")
    print(f"  日内动量IC t检验: t={t_id:.3f}, p={p_id:.4f}")

    all_ic_overnight.extend(period_ic_overnight)
    all_ic_intraday.extend(period_ic_intraday)

print(f"\n{'='*70}\n 三段区间汇总\n{'='*70}")
on_arr = np.array(all_ic_overnight)
id_arr = np.array(all_ic_intraday)
print(f"  隔夜动量IC(全部区间汇总): 均值={on_arr.mean():.4f}, n={len(on_arr)}")
print(f"  日内动量IC(全部区间汇总): 均值={id_arr.mean():.4f}, n={len(id_arr)}")
print("\n全部完成。")
