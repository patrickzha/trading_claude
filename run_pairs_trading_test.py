"""
配对交易引擎在标普500(点位时间正确的554只票池)上的首次验证。跟现有
周频动量策略完全独立，用的是同一份价格数据(point_in_time_universe_cache.pkl)
但是全新的交易逻辑(pairs_trading.py)。

测几组参数(入场z-score阈值、每个行业选几对)，看有没有一组能跑出统计上
站得住脚的正收益——如果配对交易这条路也测不出edge，至少证明"标普500
这个池子本身在能公开获取的数据上普遍缺乏可利用的定价效率缺口"这个更
宏观的判断，不只是动量这一种打法不行。
"""
from __future__ import annotations

import pickle
import time

from pairs_trading import backtest_pairs, metrics_from_periods

CACHE_PATH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad/point_in_time_universe_cache.pkl"

CONFIGS = {
    "default(entry2.0_top2)": {"entry_z": 2.0, "top_n_per_sector": 2},
    "entry1.5_top2": {"entry_z": 1.5, "top_n_per_sector": 2},
    "entry2.5_top2": {"entry_z": 2.5, "top_n_per_sector": 2},
    "entry2.0_top4": {"entry_z": 2.0, "top_n_per_sector": 4},
    "entry2.0_top1": {"entry_z": 2.0, "top_n_per_sector": 1},
}


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    sector_map = cache["sector_map"]
    print(f"数据: {len(price_df.columns)}只票, {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    for name, kwargs in CONFIGS.items():
        t0 = time.time()
        period_results, all_trades = backtest_pairs(price_df, sector_map, **kwargs)
        m = metrics_from_periods(period_results)
        elapsed = time.time() - t0
        print(f"\n--- {name} ({kwargs}) ---")
        print(f"  周期数={m['n_periods']}, 总交易笔数={len(all_trades)}, "
              f"平均每期交易数={m.get('avg_n_trades_per_period', 0):.1f}")
        if m['CAGR'] is not None:
            print(f"  CAGR={m['CAGR']*100:.2f}%, Sharpe={m['Sharpe']:.3f}, MaxDD={m['MaxDD']*100:.2f}%")
        else:
            print("  周期数不足，无法计算指标")
        print(f"  耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
