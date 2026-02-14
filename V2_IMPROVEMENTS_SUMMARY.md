# V2 改进总结

## 概述

根据 `Fix_neg_reward.md` 的诊断建议，本项目进行了以下关键改进来解决"Reward一直为负"的问题。

---

## 第一阶段：致命伤修复 (High Priority)

### 1. ✅ 现金仓位支持 (Cash Position)

**文件**: `environment_v2.py`, `networks_v2.py`, `config_ppo_v2.py`

**实现**:
- 增加第 472 维资产（现金）
- 现金特征为全零向量，表示零收益、零波动
- Agent 可以学习在熊市中持有现金

**配置**:
```python
trading_cfg.enable_cash = True  # 启用现金仓位
```

**效果**:
- 市场下跌时 Agent 可以将资金转入现金，避免亏损
- Reward 可以从负数回升到 0 附近

---

### 2. ✅ 调试模式 - 0交易成本 (Debug Zero Cost)

**文件**: `environment_v2.py`, `config_ppo_v2.py`, `main_ppo_v2.py`

**实现**:
- `trading_cfg.debug_zero_cost = True` 时，交易成本设为 0
- 先用 0 成本验证模型能否学会"低买高卖"

**使用**:
```bash
python main_ppo_v2.py --debug-zero-cost
```

**验证逻辑**:
- 如果 0 成本时 Reward 仍为正 → 特征无效或代码有 Bug
- 如果 0 成本能赚钱，加成本亏钱 → 换手率问题，需增加惩罚

---

## 第二阶段：奖励函数重构

### 3. ✅ 改进的奖励函数

**文件**: `environment_v2.py` 中的 `_compute_reward_v2()`

**实现**:
```python
# 风险调整奖励 = 对数收益 - 方差惩罚 - 换手率惩罚
base_reward = np.log(1 + net_return) * 100
risk_penalty = var_ret * risk_penalty_coef
turnover_penalty = turnover * turnover_penalty_coef
reward = base_reward - risk_penalty - turnover_penalty
```

**奖励模式**:
| 模式 | 说明 | 适用阶段 |
|------|------|---------|
| `profit_only` | 纯收益 | 课程学习第一阶段 |
| `log_return` | 对数收益率 | 课程学习第二阶段 |
| `risk_adjusted` | 收益-风险-换手惩罚 | 正式训练 |
| `sharpe` | 原始夏普比率 | 对比实验 |

**配置**:
```python
trading_cfg.reward_mode = 'risk_adjusted'
trading_cfg.risk_penalty_coef = 10.0
trading_cfg.turnover_penalty_coef = 0.3  # 显著增加
```

---

### 4. ✅ 课程学习 (Curriculum Learning)

**文件**: `trainer_ppo_v2.py`

**实现**:
```python
curriculum_stages = [
    {'reward_mode': 'profit_only', 'cost': 0.0, 'min_updates': 30},    # 学会基本交易
    {'reward_mode': 'log_return', 'cost': 0.0, 'min_updates': 30},     # 学会风险控制
    {'reward_mode': 'risk_adjusted', 'cost': 0.0015, 'min_updates': 100}  # 学会控制换手
]
```

**使用**:
```bash
python main_ppo_v2.py --curriculum
```

**效果**:
- 分阶段增加难度，每阶段更容易收敛
- 避免一次性面对过多约束导致迷茫

---

## 第三阶段：策略分布优化

### 5. ✅ Softmax + 高斯噪声策略

**文件**: `networks_v2.py` 中的 `ActorSoftmaxGaussian`

**实现**:
- 输出基础权重 `mu` = Softmax(logits)
- 在 logits 层添加高斯噪声进行探索
- 支持 Top-K 截断（只保留前 K 只股票）

**对比**:
| 特性 | Dirichlet | Softmax+Gaussian |
|------|-----------|------------------|
| 高维采样 | 困难 | 容易 |
| 稀疏权重 | 难产生 | 容易（Top-K）|
| 梯度方差 | 大 | 较小 |
| 数值稳定性 | 一般 | 更好 |

**使用**:
```bash
python main_ppo_v2.py --policy-type softmax_gaussian --top-k 10
```

---

## 新增文件说明

| 文件 | 说明 |
|------|------|
| `environment_v2.py` | 支持现金仓位和改进奖励的交易环境 |
| `config_ppo_v2.py` | V2 配置（课程学习、调试模式等） |
| `networks_v2.py` | V2 网络（Softmax+Gaussian 策略） |
| `ppo_agent_v2.py` | V2 PPO Agent |
| `trainer_ppo_v2.py` | V2 训练器（支持课程学习） |
| `main_ppo_v2.py` | V2 主程序入口 |
| `run_v2.sh` | 快速启动脚本 |
| `README_v2.md` | V2 使用文档 |

---

## 推荐使用流程

### Step 1: 调试验证（必须！）
```bash
./run_v2.sh debug
# 或
python main_ppo_v2.py --debug-zero-cost --enable-cash --reward-mode profit_only --max-windows 2
```

**检查点**:
- Reward 是否从负数上升到 0 附近？
- Cash weight 是否合理（20-80% 之间波动）？
- 如果没有，检查特征或降低 batch size

### Step 2: 加入风险控制
```bash
./run_v2.sh zero-cost-test
```

### Step 3: 课程学习
```bash
./run_v2.sh curriculum
```

### Step 4: 完整训练
```bash
./run_v2.sh full
```

---

## 关键参数速查

### 如果 Reward 仍为负
```bash
# 降低学习率
--lr 0.0001

# 减小 batch size
--batch-size 256

# 使用最简单的奖励
--reward-mode profit_only --debug-zero-cost
```

### 如果换手率过高
```bash
# 增加换手率惩罚
--turnover-penalty 0.5

# 限制持仓数量
--top-k 10

# 启用现金仓位（自动降低换手）
--enable-cash
```

### 如果波动过大
```bash
# 增加风险惩罚
--risk-penalty 20.0

# 使用对数收益
--reward-mode log_return
```

---

## 与原版本的兼容性

- V2 版本是独立的，不修改原文件
- 可以并行运行 V1 和 V2 进行对比
- 数据格式兼容，使用相同的 `processed_data.csv`

---

## 预期改进效果

| 指标 | V1 (原版本) | V2 (改进版) |
|------|------------|------------|
| Train Reward | 通常为负 | 应能转为正 |
| Test Sharpe | < 0 | 目标 > 0.5 |
| 换手率 | 极高 | 应降低 50%+ |
| 现金仓位 | N/A | 20-50% |
| 收敛速度 | 慢或不收敛 | 30-50 updates |

---

## 故障排除

### Q: 安装依赖？
V2 使用与 V1 相同的依赖，无需额外安装。

### Q: 显存不足？
```bash
--batch-size 256  # 从 512 减小
```

### Q: 如何对比 V1 和 V2？
```bash
# 运行 V1
python main_ppo.py

# 运行 V2
python main_ppo_v2.py --debug-zero-cost --enable-cash
```

---

## 参考文档

- `Fix_neg_reward.md` - 原始诊断报告
- `README_v2.md` - V2 详细使用文档
- `PPO_USAGE.md` - PPO 算法说明
