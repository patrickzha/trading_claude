"""
Cycle65真正的新框架尝试(用户2026-08指出之前"凑格式"的问题后，重新认真想
出来的候选)：衍生品尾部对冲——每月买入虚值看跌期权保护股票敞口，用
Black-Scholes定价、VIX作隐含波动率代理。这次会话第254条已经确认"用期权
隐含波动率偏斜做选股信号"不可行(yfinance只有当前期权链快照，没有历史
时间序列)，但这次要测的是完全不同的用法：不是拿期权数据当特征，是直接
用Black-Scholes公式模拟一个"买保护性看跌期权"操作本身的现金流，只需要
现货价格+VIX(隐含波动率代理)+无风险利率+到期时间这几个早就有的输入，
不需要真实历史期权链数据，绕开了第254条的数据可行性限制。

机制上这是一种和这次会话测过的所有风控手段都不同的东西——止损(第230+
条)是"跌破阈值就砍仓"，宏观熔断(第1/2条)是"看到宏观信号就空仓"，账户
熔断(第94/95条)是"回撤到阈值就降仓"，这次的保护性看跌期权是"提前付一笔
保险费，换取下跌超过某个点位之后的确定性保护"，本质上是买入一个凸性
payoff，不是通过择时/止损来管理风险，是直接花钱买保险。

如实标注局限：(1)VIX是标普500的隐含波动率，不是这次三元组合股票敞口
自身的隐含波动率，两者历史相关性很高但不完全相同，这是一个必要的代理；
(2)真实期权市场有buy-side/sell-side价差、流动性成本，这次假设能以
Black-Scholes理论价成交，不考虑额外摩擦成本，会让保护成本被低估。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from full_system import run_full_system
from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"

PERIODS = [
    ("2022-2026", f"{SCRATCH}/point_in_time_universe_cache.pkl",
     f"{SCRATCH}/default_full_results_cache.pkl", True),
    ("2014-2020", f"{SCRATCH}/historical_2014_2020_cache.pkl",
     f"{SCRATCH}/momentum_2014_2020_results_cache.pkl", False),
    ("2009-2014", f"{SCRATCH}/historical_2009_2014_cache.pkl",
     f"{SCRATCH}/momentum_2009_2014_results_cache.pkl", False),
]

BASE = dict(value_weighting="hrp", value_rebalance_days=5,
            value_stop_loss_pct=0.03, low_vol_stop_loss_pct=0.05)

R_FREE = 0.04  # 跟这次会话空仓现金收益率假设一致
MONTH_DAYS = 21


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def load_period(price_cache_path, mom_cache_path, is_pit):
    with open(price_cache_path, "rb") as f:
        cache = pickle.load(f)
    price_df, spy_close = cache["c"], cache.get("spy")
    vix = cache.get("vix")
    with open(mom_cache_path, "rb") as f:
        mom_results = pickle.load(f)
    if is_pit:
        membership = compute_point_in_time_membership(
            price_df.index, list(price_df.columns), cache["added_dates"], cache["removal_dates"])
    else:
        membership = {price_df.index[i].strftime("%Y-%m-%d"): list(price_df.columns) for i in range(len(price_df))}
    return price_df, spy_close, mom_results, membership, vix


def apply_protective_put(equity_nav: pd.Series, vix: pd.Series, otm_pct: float, notional_frac: float):
    """每21个交易日(约1个月)买一次覆盖notional_frac比例敞口的虚值看跌期权，
    行权价=当期起点价格*(1-otm_pct)，用起点VIX水平定价，到期时结算payoff，
    期权费用当期均摊扣除。"""
    idx = equity_nav.index
    n = len(idx)
    vix_aligned = vix.reindex(idx).ffill() / 100.0
    hedged_rets = []
    i = 0
    while i + MONTH_DAYS < n:
        S0 = equity_nav.iloc[i]
        ST = equity_nav.iloc[i + MONTH_DAYS]
        sigma = vix_aligned.iloc[i]
        if pd.isna(sigma) or sigma <= 0:
            sigma = 0.20
        K = S0 * (1 - otm_pct)
        T = MONTH_DAYS / 252.0
        premium = bs_put_price(S0, K, T, R_FREE, sigma)
        payoff = max(K - ST, 0.0)
        raw_ret = ST / S0 - 1
        # notional_frac比例的敞口买了保险：那部分敞口的净收益=原始收益-期权费(占S0比例)+期权到期payoff(占S0比例)
        hedge_contribution = notional_frac * (-premium / S0 + payoff / S0)
        hedged_ret = raw_ret + hedge_contribution
        hedged_rets.append((idx[i + MONTH_DAYS], hedged_ret))
        i += MONTH_DAYS

    dates, rets = zip(*hedged_rets)
    rets = np.array(rets)
    nav = np.cumprod(1 + rets)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9) * np.sqrt(252 / MONTH_DAYS))
    peak = np.maximum.accumulate(nav)
    mdd = float(np.min((nav - peak) / peak))
    return sharpe, mdd, nav[-1] if len(nav) else 1.0


def main():
    for label, price_cache_path, mom_cache_path, is_pit in PERIODS:
        price_df, spy_close, mom_results, membership, vix = load_period(price_cache_path, mom_cache_path, is_pit)
        print(f"\n{'='*70}\n {label}\n{'='*70}")

        r_base = run_full_system(price_df, membership, spy_close, mom_results, **BASE)
        equity_nav = r_base["full_nav"]
        print(f"  无对冲(基线): Sharpe={r_base['Sharpe']:.3f}, MaxDD={r_base['MaxDD']*100:.2f}%")

        if vix is None:
            print("  [警告] 这段区间没有VIX数据，跳过期权对冲测试")
            continue

        for otm_pct, notional_frac in [(0.05, 1.0), (0.10, 1.0), (0.05, 0.5)]:
            sharpe, mdd, final_nav = apply_protective_put(equity_nav, vix, otm_pct, notional_frac)
            print(f"  保护性看跌(OTM={otm_pct*100:.0f}%, 覆盖比例={notional_frac*100:.0f}%): "
                  f"Sharpe={sharpe:.3f}, MaxDD={mdd*100:.2f}%")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
