"""
拉取2014-06 ~ 2020-12这段跟现在(2021-2026)训练/回测完全不重叠的独立历史区间，
用当前标普500股票池（sp500_universe.json）的股票代码去拉这段更早的价格数据。

注意一个真实的局限：这里用的还是"现在的"股票池名单去拉更早的历史，一部分
现在在标普500里的公司2014-2020年可能还没上市/还没被纳入指数（比如2020年
上市的公司在这段区间根本没有数据）——这跟"验证结论"第13条记录的生存偏差
是同一类问题，不重新展开，缺数据的票会在下面被跳过，不强求。

这段数据的用途：不是训练新的模型或者继续调参，是给已经采纳的三个改进
（止损收紧2%、RSI阈值放开到100、样本新近度加权26周）做一次独立的样本外
复核——2014-2020跟2021-2026的市场结构很不一样（这段包含2015-16年的
盘整/小熊市、2018年Q4暴跌、2020年3月新冠暴跌），如果这三个改进在这段
完全没用来调过参的历史上也站得住，说明它们更可能是真实规律而不是过拟合
到2021年之后特定的市场环境。
"""
from __future__ import annotations

import pickle
import time

import pandas as pd
import yfinance as yf

from us_stock_screener import SECTOR_MAP

OUTPUT_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/historical_2014_2020_cache.pkl"

START = "2014-06-01"
END = "2021-01-01"


def main():
    tickers = list(SECTOR_MAP.keys())
    print(f"拉取 {len(tickers)} 只票的历史数据（{START} ~ {END}）...")
    t0 = time.time()

    all_tickers = tickers + ["^VIX"]
    data = yf.download(all_tickers, start=START, end=END, progress=False,
                       group_by="ticker", auto_adjust=True)

    close_dict, open_dict, high_dict, low_dict, vol_dict = {}, {}, {}, {}, {}
    skipped = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[t].dropna()
            else:
                df_t = data.dropna()
            if len(df_t) > 250:  # 至少要有约1年数据，太短的（比如2020年才上市）直接跳过
                close_dict[t] = df_t["Close"]
                open_dict[t] = df_t["Open"]
                high_dict[t] = df_t["High"]
                low_dict[t] = df_t["Low"]
                vol_dict[t] = df_t["Volume"]
            else:
                skipped.append(t)
        except Exception:
            skipped.append(t)
            continue

    c_df = pd.DataFrame(close_dict).dropna(how="all")
    o_df = pd.DataFrame(open_dict).dropna(how="all")
    h_df = pd.DataFrame(high_dict).dropna(how="all")
    l_df = pd.DataFrame(low_dict).dropna(how="all")
    v_df = pd.DataFrame(vol_dict).dropna(how="all")

    try:
        if isinstance(data.columns, pd.MultiIndex):
            vix = data["^VIX"]["Close"]
        else:
            vix = data["Close"] if "^VIX" in data.columns else None
    except Exception:
        vix = None
    if vix is None or vix.empty:
        vix = pd.Series(20, index=c_df.index)

    common_idx = c_df.index.intersection(o_df.index).intersection(v_df.index)
    c_df, o_df, h_df, l_df, v_df = [df.loc[common_idx] for df in [c_df, o_df, h_df, l_df, v_df]]
    vix = vix.reindex(common_idx).ffill()

    print(f"有效股票: {len(c_df.columns)}/{len(tickers)}（跳过 {len(skipped)} 只数据不足250天的，"
          f"多数是2020年之后才在这段区间上市/有数据的票，符合预期）")
    print(f"交易日: {len(common_idx)}, {common_idx[0].date()} ~ {common_idx[-1].date()}")

    try:
        spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
        spy_close = spy["Close"]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        spy_close = spy_close.reindex(common_idx).ffill()
    except Exception:
        spy_close = None

    cache = {"c": c_df, "o": o_df, "h": h_df, "l": l_df, "v": v_df, "vix": vix, "spy": spy_close}
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"耗时: {(time.time()-t0)/60:.1f}分钟")
    print(f"已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
