"""
Cycle24大修大补①：第100条的SHAP稳定性测试跟第108条揭示的问题一样，
调用get_ai_weekly_picks时没有做纳入日期过滤，候选池混入了"未来才被纳入
标普500"的票——用第108条的sp500_membership_added_dates.json，把三段
区间决策日当天的候选池限制到点位时间正确的票，重新跑一次SHAP稳定性
测试，看特征重要性排名的跨区间稳定性结论是否有变化。
"""
from __future__ import annotations

import io
import json
import pickle
import re
from contextlib import redirect_stdout

import pandas as pd

from us_stock_screener import get_ai_weekly_picks
from scipy.stats import spearmanr

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
ADDED_DATES_PATH = "/Users/zhang/Desktop/trading/sp500_membership_added_dates.json"

with open(ADDED_DATES_PATH) as f:
    added_dates_raw = json.load(f)
added_dates = {t: pd.Timestamp(d) for t, d in added_dates_raw.items()}

periods = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl"),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl"),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl"),
]

rankings = {}

for label, price_cache_path in periods:
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    c_df, o_df, h_df, l_df, v_df, vix = cache["c"], cache["o"], cache["h"], cache["l"], cache["v"], cache["vix"]
    sector_map = cache.get("sector_map", {})

    decision_date = c_df.index[-30]
    valid_tickers = [t for t in c_df.columns if t in added_dates and added_dates[t] <= decision_date]
    dropped = len(c_df.columns) - len(valid_tickers)
    print(f"\n=== {label}: 决策日={decision_date.date()}, 剔除{dropped}/{len(c_df.columns)}只未纳入的票 ===")

    c_df_f = c_df[valid_tickers]
    o_df_f = o_df[valid_tickers]
    h_df_f = h_df[valid_tickers]
    l_df_f = l_df[valid_tickers]
    v_df_f = v_df[valid_tickers]
    sector_map_f = {t: sector_map[t] for t in valid_tickers if t in sector_map}

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            get_ai_weekly_picks(c_df_f, o_df_f, h_df_f, l_df_f, v_df_f, vix, decision_date,
                               sector_map=sector_map_f, shap_enabled=True,
                               shap_output_path=f"{SCRATCH}/shap_corrected_{label.replace('-','_')}.png",
                               enable_earnings_blackout=False)
        except Exception as e:
            print(f"[错误] {e}")
    output = buf.getvalue()

    ranked = re.findall(r"^\s+(\S+)\s*:\s*([\d.]+)$", output, re.MULTILINE)
    if not ranked:
        print(f"  未捕获到SHAP排名输出，原始输出前500字符:\n{output[:500]}")
        continue
    ranked = [(name, float(val)) for name, val in ranked]
    rankings[label] = ranked
    print(f"  Top-8特征: {[r[0] for r in ranked[:8]]}")

if len(rankings) >= 2:
    print(f"\n{'='*20} 跨区间稳定性对比(点位时间修正后) {'='*20}")
    labels = list(rankings.keys())
    top5_sets = {l: set(name for name, _ in rankings[l][:5]) for l in labels}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            overlap = top5_sets[l1] & top5_sets[l2]
            print(f"  {l1} vs {l2}: Top-5重叠={len(overlap)}/5 {overlap}")
            names1 = [n for n, _ in rankings[l1]]
            names2 = [n for n, _ in rankings[l2]]
            common = [n for n in names1 if n in names2]
            rank1 = [names1.index(n) for n in common]
            rank2 = [names2.index(n) for n in common]
            if len(common) >= 3:
                rho, p = spearmanr(rank1, rank2)
                print(f"    完整排名Spearman相关系数(基于{len(common)}个共同特征)={rho:.3f}, p={p:.3f}")
