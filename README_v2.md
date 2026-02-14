# PPO V2 改进版 - 修复负奖励问题

本项目基于 `Fix_neg_reward.md` 的建议，针对 A 股投资组合管理任务进行了一系列关键改进。

## 核心改进

### 1. 现金仓位支持 (Cash Position)
**问题**: 原系统要求始终满仓，在熊市中必然亏损。

**解决方案**: 
- 增加第 472 维资产（现金）
- 现金收益为 0，但避免了下跌风险
- Agent 可以学习在熊市中持有现金

```python
# 启用现金仓位
python main_ppo_v2.py --enable-cash
```

### 2. 调试模式 (Debug Mode)
**问题**: 0.15% 的双边交易成本导致随机探索的 Agent 几乎必然亏损。

**解决方案**:
- `--debug-zero-cost`: 训练时使用 0 交易成本
- 先用 0 成本验证模型能否学会"低买高卖"
- 确认后再逐步增加交易成本

```python
# 调试模式（推荐首次运行）
python main_ppo_v2.py --debug-zero-cost --enable-cash --reward-mode profit_only
```

### 3. 改进的奖励函数
**问题**: Sharpe Ratio 在初期数值不稳定，负收益时对波动的惩罚变味。

**解决方案**: 多种奖励模式可选

| 模式 | 说明 | 适用阶段 |
|------|------|---------|
| `profit_only` | 纯收益 | 课程学习第一阶段 |
| `log_return` | 对数收益率（更稳定） | 课程学习第二阶段 |
| `risk_adjusted` | 收益 - 方差惩罚 - 换手惩罚 | 正式训练（推荐） |
| `sharpe` | 原始夏普比率 | 对比实验 |

```python
# 使用改进的奖励函数
python main_ppo_v2.py --reward-mode risk_adjusted --turnover-penalty 0.3 --risk-penalty 10.0
```

### 4. 课程学习 (Curriculum Learning)
**问题**: 同时学习交易逻辑、风险控制和成本控制太困难。

**解决方案**: 分阶段增加难度

```
阶段1: profit_only + 0成本     -> 学会基本交易
阶段2: log_return + 0成本      -> 学会风险控制
阶段3: risk_adjusted + 正常成本 -> 学会控制换手
```

```python
# 启用课程学习
python main_ppo_v2.py --enable-cash --curriculum
```

### 5. 改进的策略分布 (Softmax + Gaussian)
**问题**: Dirichlet 分布在 471 维空间采样困难，容易产生过于平滑或过于极端的权重。

**解决方案**: 
- 使用 Softmax 输出基础权重
- 在 Logits 层添加高斯噪声进行探索
- 支持 Top-K 截断（只保留前 K 只股票）

```python
# 使用改进的策略分布
python main_ppo_v2.py --policy-type softmax_gaussian --top-k 10
```

## 快速开始

### 第一步：调试模式验证（关键！）

```bash
# 先用最简单的设置验证系统是否能学到正收益
python main_ppo_v2.py \
    --debug-zero-cost \
    --enable-cash \
    --reward-mode profit_only \
    --max-windows 2 \
    --batch-size 256
```

**期望结果**: Reward 应该从负数逐渐上升到接近 0 或正值。
- 如果 Reward 仍为正：问题在交易逻辑，检查特征或模型
- 如果 Reward 为正：说明模型有效，可以增加难度

### 第二步：逐步增加难度

```bash
# 加入风险控制，但仍保持0成本
python main_ppo_v2.py \
    --debug-zero-cost \
    --enable-cash \
    --reward-mode log_return \
    --max-windows 2
```

### 第三步：加入交易成本

```bash
# 使用课程学习自动分阶段
python main_ppo_v2.py \
    --enable-cash \
    --curriculum \
    --transaction-cost 0.0015
```

### 第四步：完整训练

```bash
# 完整训练（所有改进都启用）
python main_ppo_v2.py \
    --enable-cash \
    --reward-mode risk_adjusted \
    --turnover-penalty 0.3 \
    --risk-penalty 10.0 \
    --policy-type softmax_gaussian \
    --transaction-cost 0.0015
```

## 参数说明

### 调试相关
- `--debug-zero-cost`: 训练时使用 0 交易成本
- `--debug-mode`: 启用详细日志
- `--max-windows N`: 只训练前 N 个窗口（快速测试）

### 现金仓位
- `--enable-cash`: 启用现金仓位（默认开启）
- `--no-cash`: 禁用现金仓位

### 奖励函数
- `--reward-mode {profit_only,log_return,sharpe,risk_adjusted}`: 奖励模式
- `--reward-scale FLOAT`: 奖励缩放系数（默认 1.0）
- `--turnover-penalty FLOAT`: 换手率惩罚系数（默认 0.3，建议 0.1-0.5）
- `--risk-penalty FLOAT`: 风险（方差）惩罚系数（默认 10.0）

### 课程学习
- `--curriculum`: 启用课程学习

### 策略分布
- `--policy-type {dirichlet,softmax_gaussian}`: 策略分布类型
- `--top-k INT`: Top-K 截断数量（如 10 表示只保留前 10 只股票）

### 交易成本
- `--transaction-cost FLOAT`: 双边交易成本（默认 0.0015 = 0.15%）

### 训练控制
- `--batch-size INT`: PPO batch size（默认 512）
- `--lr FLOAT`: 学习率（默认 0.0003）

## 关键参数调优建议

### 如果 Reward 仍为负

1. **检查特征**: 运行监督学习模型验证特征是否包含预测信息
2. **降低 batch size**: 尝试 `--batch-size 256` 或更小
3. **增加换手率惩罚**: 尝试 `--turnover-penalty 0.5` 抑制过度交易
4. **启用调试模式**: 先用 `--debug-zero-cost` 确认模型能工作

### 如果换手率过高

1. **增加换手率惩罚**: `--turnover-penalty 0.5` 甚至更高
2. **使用 Top-K**: `--top-k 10` 限制持仓数量
3. **启用现金仓位**: Agent 可以持有现金而不是频繁换仓

### 如果收益波动过大

1. **增加风险惩罚**: `--risk-penalty 20.0`
2. **使用 log_return 奖励**: `--reward-mode log_return`

## 文件结构

```
exper_rl/
├── environment_v2.py       # V2: 支持现金仓位和改进奖励
├── config_ppo_v2.py        # V2: 课程学习和调试配置
├── networks_v2.py          # V2: Softmax+Gaussian策略
├── ppo_agent_v2.py         # V2: 改进的PPO Agent
├── trainer_ppo_v2.py       # V2: 课程学习训练器
├── main_ppo_v2.py          # V2: 主程序入口
├── run_v2.sh               # V2: 快速启动脚本
└── README_v2.md            # 本文档
```

## 与 V1 的对比

| 特性 | V1 (原版本) | V2 (改进版) |
|------|------------|------------|
| 现金仓位 | ❌ 不支持 | ✅ 支持 |
| 调试模式 | ❌ 无 | ✅ 0成本调试 |
| 奖励函数 | Sharpe Only | 多种模式可选 |
| 课程学习 | ❌ 无 | ✅ 支持 |
| 策略分布 | Dirichlet | Dirichlet + Softmax+Gaussian |
| Top-K 截断 | ❌ 无 | ✅ 支持 |

## 预期效果

按照建议的使用顺序：

1. **调试模式**: Reward 应该能在 50 个 update 内上升到 0 附近
2. **课程学习**: 每个阶段应该能在 30-50 个 update 内收敛
3. **完整训练**: 预期 Test Sharpe > 0.5，平均现金仓位在 20-50% 之间

## 故障排除

### Q: 仍然出现 NaN/Inf
A: 尝试降低学习率 `--lr 0.0001` 或减小 batch size `--batch-size 256`

### Q: Agent 始终 100% 现金
A: 检查奖励函数设置，确保收益奖励足够大。尝试 `--reward-mode profit_only` 先验证交易逻辑。

### Q: 换手率仍然过高
A: 显著增加惩罚 `--turnover-penalty 0.5`，或启用 Top-K `--top-k 5`

### Q: 验证集表现好但测试集差
A: 可能是过拟合，尝试增加 dropout 或减小 n_epochs

## 参考

- `Fix_neg_reward.md`: 原始诊断和建议文档
- `PPO_USAGE.md`: PPO 算法使用说明
