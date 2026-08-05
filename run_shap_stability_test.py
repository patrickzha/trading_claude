"""
Cycle20大修大补②：SHAP特征重要性此前只是训练时的诊断工具(每次单独看一眼)，
从没有专门跨三段完全独立的历史区间比较过"模型认为重要的特征"是否稳定。
如果模型在不同区间依赖的特征差异很大(比如2009-2014主要看RSI，2014-2020
主要看成交量，2022-2026主要看波动率)，说明模型学到的更多是那段历史特定
的噪声模式，不是一个稳定、可泛化的规律——这是对"验证结论"第29条(pred_ret
跟net_ret相关系数接近0，模型排序能力接近噪声)的一个新角度补充诊断，用
"模型自己认为在依赖什么"而不是"模型的预测对不对"来看问题。

方法：每段区间挑一个有完整历史回看窗口的代表性决策日，跑一次
get_ai_weekly_picks(shap_enabled=True)，捕获打印出来的完整特征重要性
排名，比较三段的Top-5特征重叠度和整体排名的Spearman相关系数。
"""
from __future__ import annotations

import io
import pickle
import re
from contextlib import redirect_stdout

from us_stock_screener import get_ai_weekly_picks
from scipy.stats import spearmanr

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

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

    # 挑区间末尾往前推一点的决策日(留出验证折)，保证有完整训练窗口
    decision_date = c_df.index[-30]
    print(f"\n=== {label}: 决策日={decision_date.date()} ===")

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            get_ai_weekly_picks(c_df, o_df, h_df, l_df, v_df, vix, decision_date,
                               sector_map=sector_map, shap_enabled=True,
                               shap_output_path=f"{SCRATCH}/shap_{label.replace('-','_')}.png",
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
    print(f"\n{'='*20} 跨区间稳定性对比 {'='*20}")
    labels = list(rankings.keys())
    top5_sets = {l: set(name for name, _ in rankings[l][:5]) for l in labels}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            overlap = top5_sets[l1] & top5_sets[l2]
            print(f"  {l1} vs {l2}: Top-5重叠={len(overlap)}/5 {overlap}")
            # Spearman：用共同特征在两段区间里的排名做相关系数
            names1 = [n for n, _ in rankings[l1]]
            names2 = [n for n, _ in rankings[l2]]
            common = [n for n in names1 if n in names2]
            rank1 = [names1.index(n) for n in common]
            rank2 = [names2.index(n) for n in common]
            if len(common) >= 3:
                rho, p = spearmanr(rank1, rank2)
                print(f"    完整排名Spearman相关系数(基于{len(common)}个共同特征)={rho:.3f}, p={p:.3f}")
