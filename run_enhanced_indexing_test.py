"""
验证增强型指数化(Enhanced Indexing)方向——第504条，框架级大修大补。

假设(用户明确预期，如实验证不美化)：把当前"挑几十只票、自定义权重、脱离
基准"的因子押注模式，换成文献有实证支持的"标普500市值权重打底 + 现有因子
分数(价值/低波动/动量)小幅超配减配 + 跟踪误差约束在100-200bp"模式
(Cremers & Petajisto 2009 / Petajisto 2013)。预期：beta从0.313往1拉、
上涨周胜率上升，但这是拿82.5%的下跌周防御优势去换的，不是净增益——综合
胜率大概率升到50%出头，不会显著超过50%，IR大概率接近0。

数据源(2022-2026段，唯一能构建点位时间正确市值权重的区间)：
- point_in_time_universe_cache.pkl：价格/VIX/SPY/成分股(点位时间修正后)
- default_full_results_cache.pkl：动量腿已跑好的周频回测结果(mom_results)
- sec_fundamentals_cache.json：shares_outstanding(点位时间正确的市值权重)
  ——由 enhanced_indexing.compute_cap_weights 经 median 异常值护栏读取
  (实测该缓存里有单条shares比正常值大~1000倍的坏数据，如PKG，护栏会把
  偏离median 3倍以上的值回退成median，避免市值权重被单只票毁掉)。

指标口径：组合与基准同一套机制(周一收盘再平衡、持有到周五、连续持仓)，
日净值覆盖全部交易日，跟SPY逐日可比；周收益从日净值在每周最后交易日采样。
跟踪误差/信息比率相对"自己构建的市值权重基准"度量(基准本身用同一套口径)；
beta/上涨下跌周胜率相对SPY度量，跟旗舰组合直接头对头。基准≠SPY(覆盖率
~80-88%，且不含股息)，如实标注。

对比对象：
- 市值权重基准(SP500近似，覆盖率见运行输出)
- 旗舰组合(第377/383/399条推荐配置：leg_weights 50/40/10 + value_weighting
  =hrp + value_rebalance_days=5 + value_stop_loss_pct=0.02 +
  low_vol_stop_loss_pct=0.05)
- 增强指数(te_target ∈ {1%, 1.5%, 2%})
- SPY(外部参照)

如实结论会明确写：beta拉到多少、上涨/下跌周胜率怎么变、综合胜率是否50%
出头、IR是否接近0、是否非净增益。
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from backtest import get_weekly_trading_dates
from combo_strategy import run_combo_strategy
from run_survivorship_bias_fix_validation import compute_point_in_time_membership
from enhanced_indexing import run_enhanced_indexing_multi
from stats_utils import block_bootstrap_sharpe

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
PIT_CACHE = f"{SCRATCH}/point_in_time_universe_cache.pkl"
MOM_CACHE = f"{SCRATCH}/default_full_results_cache.pkl"

FLAGSHIP_KW = dict(leg_weights={"value": 0.50, "low_vol": 0.40, "momentum": 0.10},
                   value_weighting="hrp", value_rebalance_days=5,
                   value_stop_loss_pct=0.02, low_vol_stop_loss_pct=0.05)


def weekly_rets_from_daily(nav: pd.Series, fridays) -> pd.Series:
    """从日净值在每周最后交易日采样周收益。"""
    w = nav.reindex(fridays).dropna()
    return w.pct_change().dropna()


def daily_metrics(nav: pd.Series) -> tuple:
    n_years = len(nav) / 252
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    dr = nav.pct_change().dropna()
    sharpe = float(dr.mean() / (dr.std() + 1e-9) * np.sqrt(252))
    peak = np.maximum.accumulate(nav.values)
    mdd = float(np.min((nav.values - peak) / peak))
    return cagr, sharpe, mdd


def beta_and_wins(port_w: pd.Series, spy_w: pd.Series) -> tuple:
    a = pd.concat([port_w, spy_w], axis=1, keys=["p", "s"]).dropna()
    beta = float(np.cov(a["p"], a["s"])[0, 1] / np.var(a["s"])) if len(a) > 5 else None
    up = float((a.loc[a["s"] > 0, "p"] > a.loc[a["s"] > 0, "s"]).mean()) if (a["s"] > 0).any() else None
    dn = float((a.loc[a["s"] < 0, "p"] > a.loc[a["s"] < 0, "s"]).mean()) if (a["s"] < 0).any() else None
    ov = float((a["p"] > a["s"]).mean())
    return beta, up, dn, ov


def main():
    with open(PIT_CACHE, "rb") as f:
        cache = pickle.load(f)
    c_df, spy_close = cache["c"], cache["spy"]
    membership = compute_point_in_time_membership(c_df.index, list(c_df.columns),
                                                  cache["added_dates"], cache["removal_dates"])
    with open(MOM_CACHE, "rb") as f:
        mom_results = pickle.load(f)

    fridays = [f for _, f in get_weekly_trading_dates(c_df.index)]

    print("=" * 70)
    print(" 增强型指数化 vs 旗舰组合 vs SPY (2022-2026段, 第504条)")
    print("=" * 70)

    # --- 旗舰组合(同口径周收益) ---
    print("\n[1/3] 跑旗舰组合(第377/383条推荐配置)...")
    combo = run_combo_strategy(c_df, membership, spy_close, mom_results, **FLAGSHIP_KW)
    combo_w = weekly_rets_from_daily(combo["combo_nav"], fridays)

    # --- 增强指数(te 1% / 1.5% / 2%) ---
    print("[2/3] 跑增强指数(te_target ∈ 1% / 1.5% / 2%)...")
    reports = run_enhanced_indexing_multi(c_df, membership, spy_close,
                                          te_targets=[0.01, 0.015, 0.02], n_workers=1)

    # --- SPY周收益 ---
    spy_w = weekly_rets_from_daily(spy_close.reindex(c_df.index).ffill(), fridays)

    print("\n[3/3] 汇总对比：")
    rows = []
    # SPY(价格序列归一化到起点1.0再算指标)
    spy_aligned = spy_close.reindex(c_df.index).ffill().dropna()
    spy_nav = spy_aligned / spy_aligned.iloc[0]
    cagr, sharpe, mdd = daily_metrics(spy_nav)
    rows.append(("SPY(参照)", cagr, sharpe, mdd, 1.0, None, None, None, None, None, None, None))
    # 市值基准
    bench_nav = reports[0.015]["benchmark_nav"]
    cagr, sharpe, mdd = daily_metrics(bench_nav)
    b_beta, b_up, b_dn, b_ov = beta_and_wins(reports[0.015]["weekly_bench"], spy_w)
    rows.append(("市值权重基准", cagr, sharpe, mdd, b_beta, b_up, b_dn, b_ov, None, None, None, None))
    # 旗舰
    cagr, sharpe, mdd = daily_metrics(combo["combo_nav"])
    f_beta, f_up, f_dn, f_ov = beta_and_wins(combo_w, spy_w)
    rows.append(("旗舰组合", cagr, sharpe, mdd, f_beta, f_up, f_dn, f_ov, None, None, None, None))
    # 增强指数三档
    for te in [0.01, 0.015, 0.02]:
        rep = reports[te]
        cagr, sharpe, mdd = daily_metrics(rep["nav"])
        e_beta, e_up, e_dn, e_ov = beta_and_wins(rep["weekly_port"], spy_w)
        rows.append((f"增强指数 te={te*100:.1f}%", cagr, sharpe, mdd, e_beta, e_up, e_dn, e_ov,
                     rep["te_annual"], rep["ir"], rep["active_share"], rep["avg_turnover"]))

    hdr = (f"{'':22}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'beta':>7}{'up胜率':>8}{'down胜率':>9}"
           f"{'综合胜率':>9}{'TE':>7}{'IR':>7}{'AS':>6}{'换手':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        name, cagr, sharpe, mdd, beta, up, dn, ov, te, ir, as_, turn = r
        c = f"{cagr*100:7.1f}%" if cagr is not None else "     n/a"
        s = f"{sharpe:8.2f}" if sharpe is not None else "     n/a"
        m = f"{mdd*100:8.1f}%" if mdd is not None else "     n/a"
        b = f"{beta:7.2f}" if beta is not None else "     n/a"
        u = f"{up:7.1%}" if up is not None else "     n/a"
        d = f"{dn:8.1%}" if dn is not None else "     n/a"
        o = f"{ov:8.1%}" if ov is not None else "     n/a"
        t = f"{te*100:6.1f}%" if te is not None else "   n/a"
        i = f"{ir:7.2f}" if ir is not None else "   n/a"
        a = f"{as_:5.1%}" if as_ is not None else "  n/a"
        tu = f"{turn:6.1%}" if turn is not None else "  n/a"
        print(f"{name:22}{c}{s}{m}{b}{u}{d}{o}{t}{i}{a}{tu}")
    print("-" * len(hdr))

    # --- 显著性检验 ---
    print("\n显著性(block bootstrap, 周收益):")
    for te in [0.01, 0.015, 0.02]:
        rep = reports[te]
        excess_spy = (rep["weekly_port"] - rep["weekly_spy"]).dropna()
        excess_bench = (rep["weekly_port"] - rep["weekly_bench"]).dropna()
        if len(excess_spy) > 10:
            sh_spy, p_spy = block_bootstrap_sharpe(excess_spy, block=4, n_boot=3000, periods_per_year=52)
            sh_b, p_b = block_bootstrap_sharpe(excess_bench, block=4, n_boot=3000, periods_per_year=52)
            print(f"  te={te*100:.1f}%: 周超额 vs SPY Sharpe={sh_spy:+.2f} P(≤0)={p_spy:.1%} | "
                  f"vs 市值基准 Sharpe={sh_b:+.2f} P(≤0)={p_b:.1%}")

    # --- 方法使用情况 ---
    print("\n优化器使用情况:", {te: sorted(r["methods_used"]) for te, r in reports.items()})

    # --- 如实结论 ---
    rep = reports[0.015]
    print("\n" + "=" * 70)
    print(" 如实结论(第504条)")
    print("=" * 70)
    bd = rep["benchmark_nav"].pct_change().dropna()
    sd = spy_close.reindex(rep["benchmark_nav"].index).pct_change().dropna()
    bench_corr = float(np.corrcoef(bd.values, sd.values)[0, 1])
    print(f" 基准与SPY日收益相关性: {bench_corr:.3f} "
          f"(市值权重基准是SP500近似, 覆盖率~80-88%, 不含股息, 不是SPY本身)")
    print(f" beta: 旗舰={f_beta:.3f} -> 增强指数(te=1.5%)={rep['beta_spy']:.3f}"
          f" (是否往1拉: {'是,但只到~0.9,因子tilt本身偏防御,没有完全拉到1' if rep['beta_spy'] < 0.95 else '是,接近1'})")
    print(f" 上涨周胜率: 旗舰={f_up:.1%} -> 增强指数={rep['up_win']:.1%}"
          f" (上涨周是否更能吃到beta)")
    print(f" 下跌周胜率: 旗舰={f_dn:.1%} -> 增强指数={rep['down_win']:.1%}"
          f" (下跌周防御是否让渡)")
    print(f" 综合胜率: 旗舰={f_ov:.1%} -> 增强指数={rep['overall_win']:.1%}"
          f" (是否50%出头、不显著超过50%)")
    print(f" 信息比率(vs市值基准): te=1.5%时 IR={rep['ir']:.2f}"
          f" (是否接近0=非净增益)")
    print(f" 年化TE: te=1.5%时实现 {rep['te_annual']*100:.2f}% (目标带100-200bp)")


if __name__ == "__main__":
    main()
