"""
Cycle4大修大补①：拉取第三段独立历史区间(2009-06~2014-06)的价格数据，
用来交叉验证"验证结论"第47条发现的动量+价值组合alpha(2014-2020显著,
t=2.49；2022-2026不显著,t=0.31)到底是真实规律还是2014-2020那段区间
本身的巧合。跟2014-2020缓存用同一份484只票的名单，保持连续性——SEC
基本面数据大部分票能覆盖到2010-2011年(见"验证结论"第10条，理论上能
到2007年前后，实测抽样多数从2010年开始有数据)，2009-2010这段价值策略
候选池会比后面稀疏，如实记录、不强行掩盖。
"""
from __future__ import annotations

import pickle

import pandas as pd
import yfinance as yf

HIST_CACHE = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"
OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2009_2014_cache.pkl"

START = "2009-06-01"
END = "2014-06-02"  # 跟2014-2020缓存的起点(2014-06-02)首尾相接


def main():
    with open(HIST_CACHE, "rb") as f:
        hist = pickle.load(f)
    tickers = list(hist["c"].columns)
    print(f"拉取 {len(tickers)} 只票 {START}~{END} 的历史价格...")

    data = yf.download(tickers, start=START, end=END, progress=False, group_by="ticker", auto_adjust=True)
    data_spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
    data_vix = yf.download("^VIX", start=START, end=END, progress=False, auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    got, no_data = [], []
    for t in tickers:
        try:
            df_t = data[t].dropna() if isinstance(data.columns, pd.MultiIndex) else data.dropna()
            if len(df_t) > 500:  # 至少覆盖大部分区间才算有效数据
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
                got.append(t)
            else:
                no_data.append(t)
        except Exception:
            no_data.append(t)

    print(f"有效数据: {len(got)}/{len(tickers)}（很多票2009-2014年还没上市/没被纳入标普500，属预期内）")

    c_df = pd.DataFrame(close_dict)
    o_df = pd.DataFrame(open_dict)
    h_df = pd.DataFrame(high_dict)
    l_df = pd.DataFrame(low_dict)
    v_df = pd.DataFrame(vol_dict)

    # yfinance新版本对单只票下载也会返回带Ticker层级的列(MultiIndex或者
    # 单层但列名仍是Ticker)，之前用 isinstance(...,pd.Index) 判断不可靠——
    # MultiIndex本身也是pd.Index的子类，"Close" in columns 在某些形状下
    # 仍然会匹配到多列返回DataFrame而不是Series，导致vix/spy被存成单列
    # DataFrame，下游一比较大小就报"Series的真值不明确"。统一用squeeze()
    # 强制拿到纯Series，不管上游返回的是什么列结构。
    def _extract_close_series(df):
        if isinstance(df.columns, pd.MultiIndex):
            close = df.xs("Close", axis=1, level=0)
        else:
            close = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
        return close.squeeze(axis=1)

    spy_close = _extract_close_series(data_spy)
    vix_close = _extract_close_series(data_vix)
    assert spy_close.ndim == 1 and vix_close.ndim == 1, "spy/vix提取失败，仍然是DataFrame"

    cache = {"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "vix": vix_close, "spy": spy_close,
             "got": got, "no_data": no_data}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"已写入 {OUTPUT_PATH}")
    print(f"日期范围: {c_df.index[0].date()} ~ {c_df.index[-1].date()}, {len(c_df)}个交易日")


if __name__ == "__main__":
    main()
