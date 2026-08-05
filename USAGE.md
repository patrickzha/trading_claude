# 怎么用这套系统（一页说明）

## 每周要做的事

```bash
.conda/bin/python3 run_weekly_paper_trading.py
```

这一条命令会：拉取实时数据 → 生成本周三条腿(动量/价值/低波动)候选 → 追加写入
`paper_trading_log.csv` → 顺带核对上一次记录的候选后来实际涨跌了多少。

**这个脚本本身不下单**，只生成信号、记录到日志。要不要真的按这个操作，是你自己的决定。

## 当前推荐配置（唯一权威来源，其余地方如有冲突以这里为准）

```python
leg_weights = {"value": 0.50, "low_vol": 0.40, "momentum": 0.10}
value_weighting = "hrp"
value_rebalance_days = 5
value_stop_loss_pct = 0.02
low_vol_stop_loss_pct = 0.05
# + 动态债券对冲层(默认IEF，加息趋势降仓)
```

这是`run_weekly_paper_trading.py`和`generate_current_recommendations.py`两个脚本
已经写死使用的配置，不需要你自己去传参数。如果想改配置做实验，改这两个脚本里
调用`run_full_system(...)`那一行的参数即可。

## 三个脚本，分别是什么

| 脚本 | 用途 |
|---|---|
| `run_weekly_paper_trading.py` | **主入口**，实时数据，每周跑，自动记录+核对历史表现 |
| `generate_current_recommendations.py` | 离线演示版，用缓存数据(不联网)，想快速看格式/调试时用 |
| `backtest.py` / `combo_strategy.py` / `full_system.py` | 回测引擎，不是日常操作用的，改代码/重新验证时才需要碰 |

## 这套系统的真实可信度（不是营销话术，是实测结论）

- 收益的主要来源是价值因子腿，动量腿基本没有独立alpha，止损机制是证据最扎实
  的部分（在正常市场和真实历史危机里都验证过）。
- **多重检验修正后，"这个配置的Sharpe是真实效应还是搜索出来的噪声"这个问题
  的诚实答案是一个24%-92%的置信区间，不是一个确定的高置信度结论**（详见
  README"现状总结"最后一段）。
- `paper_trading_log.csv`目前只有几条记录，是唯一真正的样本外证据来源，攒够
  3-6个月才有参考价值，现在还早。

## 想看"为什么"

- 当前状态、已知限制：[`README.md`](README.md)
- 500多条完整验证历史（每个参数选择的来龙去脉）：[`VALIDATION_LOG.md`](VALIDATION_LOG.md)
