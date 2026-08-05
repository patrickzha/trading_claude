"""
统计验证工具
============
1. 样本内/样本外切分：按时间顺序切，后一段从未参与过任何调参判断，
   用来检查策略是不是只在"调参用的那段历史"上好看。
2. Block bootstrap 显著性检验：周收益之间存在自相关（止损/熔断会让相邻几周
   同向），普通 iid bootstrap 会低估方差、把噪声当成显著性。这里用滑动区块
   自助法（moving block bootstrap）重采样，保留局部时序结构再算 Sharpe 的
   抽样分布。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_in_out_sample(df: pd.DataFrame, out_sample_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分为 (样本内, 样本外)，df 需已按时间升序排列。"""
    split_idx = int(len(df) * (1 - out_sample_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _sharpe(excess: np.ndarray, weeks_per_year: int) -> float:
    s = excess.std()
    return (excess.mean() / s) * np.sqrt(weeks_per_year) if s != 0 else 0.0


def block_bootstrap_sharpe_test(weekly_returns: pd.Series, rf_weekly: float,
                                weeks_per_year: int = 52,
                                n_boot: int = 5000, block_size: int = 4,
                                seed: int = 42) -> dict:
    """
    对 Sharpe 比率做 block bootstrap 抽样分布估计。

    返回:
        observed_sharpe: 实际观测到的 Sharpe
        ci_low / ci_high: bootstrap 95% 置信区间
        prob_sharpe_leq_0: bootstrap 分布中 Sharpe<=0 的比例（单侧、非严格意义
            上的 p 值，衡量"这个正的 Sharpe 有多容易只是噪声"——数值越小越
            说明这个 Sharpe 不太可能是巧合）
    """
    r = weekly_returns.to_numpy(dtype=float) - rf_weekly
    n = len(r)
    if n < block_size * 2:
        return {
            "n_obs": n, "n_boot": 0, "observed_sharpe": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "prob_sharpe_leq_0": np.nan,
        }

    observed = _sharpe(r, weeks_per_year)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block_size] for s in starts])[:n]
        boot_sharpes[b] = _sharpe(sample, weeks_per_year)

    ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
    prob_leq_0 = float(np.mean(boot_sharpes <= 0))

    return {
        "n_obs": n, "n_boot": n_boot, "observed_sharpe": observed,
        "ci_low": ci_low, "ci_high": ci_high, "prob_sharpe_leq_0": prob_leq_0,
    }


def compute_period_metrics(df: pd.DataFrame, rf_weekly: float, weeks_per_year: int) -> dict:
    if len(df) == 0:
        return {"n_weeks": 0}
    years = len(df) / weeks_per_year
    final_nw = df["net_worth"].iloc[-1]
    initial_nw = df["net_worth"].iloc[0]
    cagr = (final_nw / initial_nw) ** (1 / years) - 1 if years > 0 else 0.0
    excess = df["weekly_return"].to_numpy(dtype=float) - rf_weekly
    sharpe = _sharpe(excess, weeks_per_year)
    rolling_max = df["net_worth"].cummax()
    max_dd = ((df["net_worth"] - rolling_max) / rolling_max).min()
    win_rate = (df["weekly_return"] > 0).mean()
    return {
        "n_weeks": len(df), "cagr": cagr, "sharpe": sharpe,
        "max_dd": max_dd, "win_rate": win_rate,
    }


def print_statistical_validation(df: pd.DataFrame, rf_weekly: float,
                                 weeks_per_year: int = 52,
                                 out_sample_frac: float = 0.2) -> None:
    """打印样本内/样本外对比 + Sharpe 显著性检验，接在主报告后面输出。"""
    in_df, out_df = split_in_out_sample(df, out_sample_frac)
    in_m = compute_period_metrics(in_df, rf_weekly, weeks_per_year)
    out_m = compute_period_metrics(out_df, rf_weekly, weeks_per_year)
    boot = block_bootstrap_sharpe_test(df["weekly_return"], rf_weekly, weeks_per_year)

    print("=" * 70)
    print(" 【统计验证：样本内 / 样本外对比】")
    print("=" * 70)
    print(f" 样本内（前{1-out_sample_frac:.0%}，{in_m.get('n_weeks', 0)}周）："
          f"CAGR {in_m.get('cagr', float('nan'))*100:.2f}%  "
          f"Sharpe {in_m.get('sharpe', float('nan')):.2f}  "
          f"最大回撤 {in_m.get('max_dd', float('nan'))*100:.2f}%  "
          f"胜率 {in_m.get('win_rate', float('nan'))*100:.1f}%")
    print(f" 样本外（后{out_sample_frac:.0%}，{out_m.get('n_weeks', 0)}周，从未用来调参）："
          f"CAGR {out_m.get('cagr', float('nan'))*100:.2f}%  "
          f"Sharpe {out_m.get('sharpe', float('nan')):.2f}  "
          f"最大回撤 {out_m.get('max_dd', float('nan'))*100:.2f}%  "
          f"胜率 {out_m.get('win_rate', float('nan'))*100:.1f}%")
    if out_m.get("n_weeks", 0) > 0 and in_m.get("sharpe") is not None:
        gap = out_m.get("sharpe", 0) - in_m.get("sharpe", 0)
        print(f" 样本外 Sharpe 相对样本内变化: {gap:+.2f}"
              f"{'（明显衰减，warning：可能是过拟合历史特定行情）' if gap < -0.3 else ''}")
    print("-" * 70)
    print(" 【Sharpe 显著性检验（block bootstrap，block=4周，n=5000）】")
    print(f" 全样本观测 Sharpe: {boot['observed_sharpe']:.2f}")
    print(f" 95% 置信区间: [{boot['ci_low']:.2f}, {boot['ci_high']:.2f}]")
    print(f" bootstrap 分布中 Sharpe<=0 的比例: {boot['prob_sharpe_leq_0']:.1%}"
          f"{'（置信区间跨过0，现在的样本量还不足以确认这个Sharpe不是噪声）' if boot['ci_low'] < 0 else '（置信区间不跨0，有统计支持）'}")
    print("=" * 70)
