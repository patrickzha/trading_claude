"""
Cycle17大修大补②验证：full_system.py里的动态债券对冲阈值(rate_trend_threshold=0.3
百分点，来自README第69条)是"看过全部历史数据后挑出来的最优值"还是"用前半段数据
选、后半段数据验证依然站得住"。这次测试量化这个参数本身的前视偏差。

方法：每段历史区间按时间对半切(前50%用来选阈值，后50%完全没被这次选择过程看到)。
- 用候选集{0.1,0.2,0.3,0.4,0.5}百分点，只在前半段上选Sharpe最高的阈值(walk-forward选择)
- 拿这个walk-forward选出来的阈值，在后半段(样本外)上算真实Sharpe
- 对照组：全历史(前+后)一起选出来的"事后最优"阈值(oracle，这就是原来选0.3pp时
  实际做的事)，同样在后半段上算Sharpe
- 另一个对照：不做任何选择，直接固定用原来定的0.3pp，在后半段上算Sharpe
如果walk-forward选择的样本外表现明显不如oracle，说明0.3pp这个值本身带前视偏差，
不能直接信任成"未来也最优"。如果三者差不多，说明这个参数选择相对稳健。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import yfinance as yf

from combo_strategy import run_combo_strategy, true_max_drawdown
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5]
BOND_WEIGHT = 0.4
BOND_WEIGHT_HIKE = 0.1
TREND_WINDOW = 60


def sharpe_of(full_ret):
    if len(full_ret) < 5:
        return float("nan")
    return float(full_ret.mean() / (full_ret.std() + 1e-9) * np.sqrt(252))


def full_ret_for_threshold(equity_ret, bond_ret, tnx_trend, threshold):
    bw = pd.Series(BOND_WEIGHT, index=equity_ret.index)
    bw[tnx_trend.reindex(equity_ret.index) > threshold] = BOND_WEIGHT_HIKE
    return ((1 - bw) * equity_ret + bw * bond_ret).dropna()


periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

for label, price_cache_path, mom_cache_path, is_pit in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)

    if is_pit:
        membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                        cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}

    equity_report = run_combo_strategy(price_df, membership, spy_close, mom_results)
    equity_nav = equity_report["combo_nav"]
    equity_ret = equity_nav.pct_change()

    start = equity_nav.index[0].strftime("%Y-%m-%d")
    end = equity_nav.index[-1].strftime("%Y-%m-%d")
    data = yf.download(["IEF", "^TNX"], start=start, end=end, progress=False,
                       group_by="ticker", auto_adjust=True)
    bond_close = data["IEF"]["Close"].dropna()
    tnx = data["^TNX"]["Close"].dropna()
    bond_ret = bond_close.reindex(equity_nav.index).ffill().pct_change()
    tnx_trend = tnx.reindex(equity_nav.index).ffill().diff(TREND_WINDOW)

    n = len(equity_ret)
    split = n // 2
    first_half_idx = equity_ret.index[:split]
    second_half_idx = equity_ret.index[split:]

    # walk-forward: 只用前半段选阈值
    wf_scores = {}
    for th in CANDIDATES:
        fr = full_ret_for_threshold(equity_ret.loc[first_half_idx], bond_ret, tnx_trend, th)
        wf_scores[th] = sharpe_of(fr)
    wf_best_th = max(wf_scores, key=wf_scores.get)

    # oracle: 用全历史选阈值(原来实际做的事)
    oracle_scores = {}
    for th in CANDIDATES:
        fr = full_ret_for_threshold(equity_ret, bond_ret, tnx_trend, th)
        oracle_scores[th] = sharpe_of(fr)
    oracle_best_th = max(oracle_scores, key=oracle_scores.get)

    # 三者在样本外(后半段)的真实表现
    wf_oos = sharpe_of(full_ret_for_threshold(equity_ret.loc[second_half_idx], bond_ret, tnx_trend, wf_best_th))
    oracle_oos = sharpe_of(full_ret_for_threshold(equity_ret.loc[second_half_idx], bond_ret, tnx_trend, oracle_best_th))
    fixed03_oos = sharpe_of(full_ret_for_threshold(equity_ret.loc[second_half_idx], bond_ret, tnx_trend, 0.3))

    print(f"\n=== {label} ===")
    print(f"  前半段选阈值(walk-forward): 各候选Sharpe={ {k: round(v,3) for k,v in wf_scores.items()} }")
    print(f"  walk-forward选出的阈值 = {wf_best_th}pp")
    print(f"  全历史选阈值(oracle/事后最优): 各候选Sharpe={ {k: round(v,3) for k,v in oracle_scores.items()} }")
    print(f"  oracle选出的阈值 = {oracle_best_th}pp")
    print(f"  --- 样本外(后半段)真实Sharpe对比 ---")
    print(f"  walk-forward阈值({wf_best_th}pp)样本外Sharpe = {wf_oos:.3f}")
    print(f"  oracle阈值({oracle_best_th}pp)样本外Sharpe   = {oracle_oos:.3f}")
    print(f"  固定0.3pp(原方案)样本外Sharpe            = {fixed03_oos:.3f}")
