"""
Cycle43大修大补①(框架改动，新数据维度——财报事件的成交量确认)：第25条
测过PEAD(用盈利惊喜幅度排序)证据偏弱，这次换一个不同的构造维度——
财报发布后的成交量异动(而不是价格/盈利惊喜幅度)作为信号。市场微观
结构文献里"放量上涨"通常被认为比"缩量上涨"更能代表真实的信息驱动
买盘(不是噪声交易)，这是同一个财报事件数据源下一个真正不同的信号
构造方式。用现成的`pead_data_cache.pkl`(471只票的历史财报日期+盈利
惊喜幅度)复用财报日期，只新增成交量异动的计算，避免重新抓取。

方法论跟第43/44条(标准化独立测试)一致：算每次财报后3日累计涨跌幅、
财报前后成交量比值(财报后3日均量/财报前20日均量)，检验"成交量异动
幅度"本身、以及"成交量异动x盈利惊喜方向"的交互项，跟接下来一周实际
收益率的相关性。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import stats

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"


def main():
    print("加载已有PEAD财报日期数据+价格/成交量缓存...")
    with open(f"{SCRATCH}/pead_data_cache.pkl", "rb") as f:
        pead_data = pickle.load(f)
    with open(f"{SCRATCH}/point_in_time_universe_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    price_df, vol_df = cache["c"], cache["v"]

    rows = []
    for ticker, events in pead_data.items():
        if ticker not in price_df.columns or ticker not in vol_df.columns:
            continue
        px = price_df[ticker]
        vol = vol_df[ticker]
        for ev in events:
            ed = ev["date"]
            if ed not in px.index:
                idx_candidates = px.index[px.index >= ed]
                if len(idx_candidates) == 0:
                    continue
                ed = idx_candidates[0]
            loc = px.index.get_loc(ed)
            if loc < 20 or loc + 8 >= len(px.index):
                continue

            pre_vol = vol.iloc[loc - 20:loc].mean()
            post_vol = vol.iloc[loc:loc + 3].mean()
            if pre_vol is None or pre_vol <= 0 or pd.isna(pre_vol) or pd.isna(post_vol):
                continue
            vol_anomaly = post_vol / pre_vol - 1

            price_reaction = px.iloc[loc + 2] / px.iloc[loc - 1] - 1 if pd.notna(px.iloc[loc + 2]) and pd.notna(px.iloc[loc - 1]) else np.nan
            fwd_ret = px.iloc[loc + 7] / px.iloc[loc + 2] - 1 if loc + 7 < len(px.index) and pd.notna(px.iloc[loc + 7]) and pd.notna(px.iloc[loc + 2]) else np.nan

            if pd.isna(price_reaction) or pd.isna(fwd_ret) or pd.isna(vol_anomaly):
                continue

            rows.append({"ticker": ticker, "date": ed, "surprise_pct": ev["surprise_pct"],
                         "vol_anomaly": vol_anomaly, "price_reaction": price_reaction, "fwd_ret": fwd_ret})

    df = pd.DataFrame(rows)
    print(f"\n共{len(df)}个财报事件样本")

    corr_vol_fwd, p_vol = stats.pearsonr(df["vol_anomaly"], df["fwd_ret"])
    print(f"\n成交量异动 vs 财报后一周前瞻收益: 相关系数={corr_vol_fwd:.4f}, p={p_vol:.4f}")

    df["confirmed"] = (df["price_reaction"] > 0) & (df["vol_anomaly"] > df["vol_anomaly"].median())
    df["unconfirmed_up"] = (df["price_reaction"] > 0) & (df["vol_anomaly"] <= df["vol_anomaly"].median())

    confirmed_fwd = df.loc[df["confirmed"], "fwd_ret"]
    unconfirmed_fwd = df.loc[df["unconfirmed_up"], "fwd_ret"]
    down_fwd = df.loc[df["price_reaction"] <= 0, "fwd_ret"]

    print(f"\n放量上涨(价格反应为正+成交量异动超过中位数): n={len(confirmed_fwd)}, 前瞻均收益={confirmed_fwd.mean()*100:.3f}%")
    print(f"缩量上涨(价格反应为正+成交量异动低于中位数): n={len(unconfirmed_fwd)}, 前瞻均收益={unconfirmed_fwd.mean()*100:.3f}%")
    print(f"财报后价格下跌: n={len(down_fwd)}, 前瞻均收益={down_fwd.mean()*100:.3f}%")

    if len(confirmed_fwd) > 5 and len(unconfirmed_fwd) > 5:
        t, p = stats.ttest_ind(confirmed_fwd, unconfirmed_fwd)
        print(f"\n放量上涨 vs 缩量上涨 t检验: t={t:.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
