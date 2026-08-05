"""
Cycle28大修大补①：把第140条验证过的最小方差组合优化框架推广到动量腿——
动量腿每周只选3只票(MAX_POS=3)，样本量比价值腿的30只小得多，用60天
历史协方差矩阵给3个资产算最小方差权重理论上噪声会更大(3x3协方差矩阵
虽然求逆更稳定，但只有3个资产，分散化空间本身就小，"能挖掘的真实风险
异质性"可能跟第141条低波动腿失败的原因类似)，但机制上完全没测过，
这次直接测，不预设结论。

用已缓存的mom_results(PIT修正后)里的picks字段(每周3只候选)，结合价格
数据重建每周的min-variance权重，代替原来隐含的等权，对比周收益。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from value_investing import _min_variance_weights

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
COV_WINDOW = 60

periods = [
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl", f"{SCRATCH}/momentum_2014_2020_results_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl", f"{SCRATCH}/momentum_2009_2014_results_cache.pkl"),
]

for label, price_cache_path, mom_cache_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)

    equal_rets, mv_rets = [], []
    for r in mom_results:
        trades = r.get("trades", [])
        if len(trades) < 2:
            continue
        tickers = [t["ticker"] for t in trades]
        trade_rets = {t["ticker"]: t["trade_ret"] for t in trades}
        decision_date = pd.Timestamp(r["date"])

        equal_ret = np.mean(list(trade_rets.values()))
        equal_rets.append(equal_ret)

        valid_tickers = [t for t in tickers if t in price_df.columns]
        if len(valid_tickers) < 2:
            mv_rets.append(equal_ret)
            continue
        hist = price_df[valid_tickers].loc[:decision_date].tail(COV_WINDOW + 1)
        rets_window = hist.pct_change().dropna()
        valid_cols = rets_window.columns[rets_window.notna().all()]
        if len(valid_cols) < 2:
            mv_rets.append(equal_ret)
            continue
        w = _min_variance_weights(rets_window[valid_cols])
        mv_ret = sum(w[t] * trade_rets[t] for t in valid_cols if t in trade_rets)
        mv_rets.append(mv_ret)

    equal_rets = np.array(equal_rets)
    mv_rets = np.array(mv_rets)
    equal_sharpe = equal_rets.mean() / (equal_rets.std() + 1e-9) * np.sqrt(52)
    mv_sharpe = mv_rets.mean() / (mv_rets.std() + 1e-9) * np.sqrt(52)
    print(f"\n=== {label} (n={len(equal_rets)}周) ===")
    print(f"  等权(现有,隐含): 周均收益={equal_rets.mean()*100:.3f}%, Sharpe={equal_sharpe:.3f}")
    print(f"  最小方差权重(新): 周均收益={mv_rets.mean()*100:.3f}%, Sharpe={mv_sharpe:.3f}")
