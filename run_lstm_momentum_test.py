"""
Cycle32大修大补②(框架改动，新模型架构)：LSTM循环神经网络(PyTorch)替代
XGBoost回归框架，测试动量选股这个场景下，一个真正不同的模型范式(时序
循环网络，显式建模"过去K周特征序列"，而不是XGBoost那种把每周特征当
独立样本、看不到跨周时序结构的树模型)能不能捕捉到XGBoost框架下测不
出来的信号。这是这次会话第一次尝试非树模型架构，属于CLAUDE.md明确
要求的"新的模型架构"这一类框架改动。

设计取舍(如实说明，不回避简化)：完整walk-forward每周重新训练LSTM
成本过高(神经网络训练比XGBoost慢一个数量级)，这里用"每13周(一个季度)
重新训练一次、每次只用截止当前的历史数据"这种粗粒度walk-forward，
仍然满足"不用到未来信息"这个核心要求，但重训练频率比主XGBoost框架
(每周)更粗，如果LSTM方向本身有效，后续可以考虑增加重训练频率换取
更高精度，这次的目标是先验证"这条路是否值得走"这个feasibility问题。

特征设计：5个滚动技术特征(1周动量/1月动量/RSI/已实现波动率比/
价格相对20日均线)，取最近8周的序列作为LSTM输入(seq_len=8, feature_dim=5)，
标签是下一周的收益率(回归目标，跟XGBoost主框架的regression目标一致，
方便直接比较)。
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_survivorship_bias_fix_validation import compute_point_in_time_membership

SCRATCH = "/private/tmp/claude-501/-Users-zhang-Desktop-trading/5d5072d6-7a4e-4d89-be2e-ee80a455235f/scratchpad"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SEQ_LEN = 8
RETRAIN_EVERY = 13
TOP_N = 10
WARMUP_WEEKS = 30


def rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_weekly_features(price_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对每只票算5个日频特征，再取每周Monday(或最近的交易日)那一行，
    返回 {feature_name: DataFrame(index=周, columns=tickers)}"""
    mom_1w = price_df.pct_change(5)
    mom_1m = price_df.pct_change(21)
    rsi = price_df.apply(rsi_series)
    ret_20 = price_df.pct_change().rolling(20).std()
    ret_60 = price_df.pct_change().rolling(60).std()
    vol_ratio = (ret_20 / ret_60.replace(0, np.nan)).fillna(1.0)
    ma20 = price_df.rolling(20).mean()
    price_to_ma20 = (price_df / ma20 - 1).fillna(0.0)

    weekly_idx = price_df.index[price_df.index.weekday == 0]
    weekly_idx = weekly_idx[weekly_idx.isin(price_df.index)]

    def to_weekly(df):
        return df.reindex(weekly_idx)

    return {
        "mom_1w": to_weekly(mom_1w),
        "mom_1m": to_weekly(mom_1m),
        "rsi": to_weekly((rsi - 50) / 50),
        "vol_ratio": to_weekly(vol_ratio - 1),
        "price_to_ma20": to_weekly(price_to_ma20),
    }, weekly_idx


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=5, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _lookup_membership(membership, membership_keys, date_str):
    """membership的key是上一次调仓决策日(通常是周五)，不是Monday本身，
    需要找<=date_str的最近key——这是这次会话从第108条PIT修正开始就
    统一使用的查找方式(见run_hrp_test.py等脚本)，这里复用同样的逻辑。"""
    valid = membership.get(date_str)
    if valid is not None:
        return valid
    candidates = [k for k in membership_keys if k <= date_str]
    if not candidates:
        return None
    return membership[max(candidates)]


def build_dataset(feat_dict, weekly_idx, price_weekly, week_start, week_end, membership, min_history=None):
    """构造 [week_start, week_end) 区间内、所有票的(序列, 标签)训练样本。
    week_end对应的标签用week_end这一周相对week_end-1的收益率(不用到week_end
    之后的数据)。min_history默认None时取模块级SEQ_LEN——不能用函数默认参数
    直接绑SEQ_LEN，因为默认参数在函数定义时就固定值了，后续如果通过
    monkeypatch改SEQ_LEN(比如序列长度敏感性测试)，默认参数不会跟着变，
    跟内部按wi-SEQ_LEN切片用的是运行时的全局值不一致，会产生长度不整齐
    的序列(第174条附近做敏感性测试时踩过这个坑)。"""
    if min_history is None:
        min_history = SEQ_LEN
    tickers = price_weekly.columns
    membership_keys = sorted(membership.keys())
    X, y = [], []
    for wi in range(max(week_start, min_history), week_end):
        date_str = weekly_idx[wi].strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            continue
        for t in tickers:
            if t not in valid:
                continue
            seq = np.stack([feat_dict[f][t].iloc[wi - SEQ_LEN:wi].values for f in feat_dict], axis=-1)
            if np.isnan(seq).any():
                continue
            label = price_weekly[t].iloc[wi] / price_weekly[t].iloc[wi - 1] - 1 if wi > 0 else np.nan
            if pd.isna(label):
                continue
            X.append(seq)
            y.append(label)
    if not X:
        return None, None
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


def train_model(X, y, epochs=15):
    model = LSTMRegressor().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X, y = X.to(DEVICE), y.to(DEVICE)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, y[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
    return model


def true_mdd(nav):
    nav = np.array(nav)
    peak = np.maximum.accumulate(nav)
    return float(np.min((nav - peak) / peak))


def main():
    with open(f"{SCRATCH}/point_in_time_universe_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    price_df = cache["c"]
    membership = compute_point_in_time_membership(price_df.index, list(price_df.columns),
                                                    cache["added_dates"], cache["removal_dates"])

    print("计算特征...")
    feat_dict, weekly_idx = build_weekly_features(price_df)
    price_weekly = price_df.reindex(weekly_idx)
    n_weeks = len(weekly_idx)
    print(f"周数: {n_weeks}, 股票数: {len(price_df.columns)}, 设备: {DEVICE}")

    lstm_weekly_rets = []
    equal_weekly_rets = []
    model = None
    last_train_wi = -999
    membership_keys = sorted(membership.keys())

    for wi in range(WARMUP_WEEKS, n_weeks - 1):
        if model is None or wi - last_train_wi >= RETRAIN_EVERY:
            print(f"  [第{wi}/{n_weeks}周] 重新训练LSTM(截止到这一周的历史数据)...")
            X, y = build_dataset(feat_dict, weekly_idx, price_weekly, 0, wi, membership)
            if X is None or len(X) < 200:
                print(f"    样本不足(X={0 if X is None else len(X)})，跳过这次重训练，沿用上一个模型")
            else:
                model = train_model(X, y)
                last_train_wi = wi
            if model is None:
                continue

        date_str = weekly_idx[wi].strftime("%Y-%m-%d")
        valid = _lookup_membership(membership, membership_keys, date_str)
        if valid is None:
            continue
        cand_tickers, seqs = [], []
        for t in price_weekly.columns:
            if t not in valid:
                continue
            seq = np.stack([feat_dict[f][t].iloc[wi - SEQ_LEN:wi].values for f in feat_dict], axis=-1)
            if np.isnan(seq).any():
                continue
            cand_tickers.append(t)
            seqs.append(seq)
        if len(cand_tickers) < TOP_N:
            continue
        X_pred = torch.tensor(np.array(seqs), dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = model(X_pred).cpu().numpy()
        top_idx = np.argsort(preds)[-TOP_N:]
        picks = [cand_tickers[i] for i in top_idx]

        fwd_rets = []
        for t in picks:
            p0, p1 = price_weekly[t].iloc[wi], price_weekly[t].iloc[wi + 1]
            if pd.notna(p0) and pd.notna(p1) and p0:
                fwd_rets.append(p1 / p0 - 1)
        if fwd_rets:
            lstm_weekly_rets.append(np.mean(fwd_rets))

        all_fwd = []
        for t in cand_tickers:
            p0, p1 = price_weekly[t].iloc[wi], price_weekly[t].iloc[wi + 1]
            if pd.notna(p0) and pd.notna(p1) and p0:
                all_fwd.append(p1 / p0 - 1)
        if all_fwd:
            equal_weekly_rets.append(np.mean(all_fwd))

    def metrics(weekly_rets):
        rets = np.array(weekly_rets)
        nav = np.cumprod(1 + rets)
        n_years = len(rets) / 52
        cagr = float(nav[-1] ** (1 / n_years) - 1) if n_years > 0 else None
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(52))
        mdd = true_mdd(nav)
        return cagr, sharpe, mdd, len(rets)

    l_cagr, l_sharpe, l_mdd, l_n = metrics(lstm_weekly_rets)
    e_cagr, e_sharpe, e_mdd, e_n = metrics(equal_weekly_rets)
    print(f"\n=== LSTM选股(top{TOP_N}) === 周数={l_n}, CAGR={l_cagr*100:.2f}%, Sharpe={l_sharpe:.3f}, MaxDD={l_mdd*100:.2f}%")
    print(f"=== 全候选池等权(基线) === 周数={e_n}, CAGR={e_cagr*100:.2f}%, Sharpe={e_sharpe:.3f}, MaxDD={e_mdd*100:.2f}%")


if __name__ == "__main__":
    main()
