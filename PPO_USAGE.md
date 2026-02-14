# PPO 算法使用指南

## 📋 算法变更说明

已从 **SAC** (Soft Actor-Critic) 切换为 **PPO** (Proximal Policy Optimization)，解决以下问题：

| 问题 | 原因 | PPO 解决方案 |
|------|------|-------------|
| 梯度消失 | SAC 的 Q-value 估计不稳定 | On-policy PPO + GAE |
| 最大熵噪声 | SAC 强制探索导致 HHI 极低 | 熵系数从 0.01 衰减至 0 |
| 收益率为 0 | 奖励信号太弱（0.01 级别） | 奖励放大 100x |
| 训练不动 | Batch size 太大导致 CUDA 错误 | Rollout buffer + mini-batch |

## 🚀 快速开始

### 1. 使用预处理数据训练（推荐）

```bash
cd /root/autodl-tmp/exper_rl

# 如果还没有预处理数据，先执行
python preprocess_data.py

# PPO 训练
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda

# 训练并对比基准
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --run-benchmarks
```

### 2. 从原始数据训练

```bash
python main_ppo.py --device cuda --run-benchmarks
```

## ⚙️ 关键超参数

### PPO 核心参数（config_ppo.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | Actor 和 Critic 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 系数，平衡偏差和方差 |
| `clip_range` | 0.2 | PPO 截断范围 |
| `entropy_coef` | 0.01 → 0 | 熵系数（线性衰减）|
| `batch_size` | 2048 | Rollout 收集步数 |
| `mini_batch_size` | 64 | 更新时的切片大小 |
| `n_epochs` | 10 | 每次收集后的更新轮数 |

### 交易参数（已优化）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `reward_scale` | 100.0 | **关键！** 收益率放大 100 倍 |
| `temperature` | 1.0 | Softmax 温度系数 |

## 🔧 工程修正

### A. 奖励缩放（Reward Scaling）
```python
# environment.py
reward_scale = 100.0  # 将日收益率从 0.01 放大到 1.0
reward = net_return * reward_scale  # 现在奖励在 [-5, +5] 范围
```

### B. Action Masking（动作掩码）
```python
# networks.py - Actor.forward()
if action_mask is not None:
    logits = logits.masked_fill(action_mask == 0, float('-1e9'))
probs = F.softmax(logits, dim=-1)  # 停牌股票权重强制为 0
```

### C. Advantage Normalization（优势归一化）
```python
# ppo_agent.py - RolloutBuffer.normalize_advantages()
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

## 📊 训练监控

### TensorBoard
```bash
tensorboard --logdir=./logs_ppo
```

### 关键指标
- **Policy Loss**: 策略损失，应逐渐下降
- **Value Loss**: 价值损失，应逐渐下降
- **Entropy**: 策略熵，应逐渐降低（从 0.01 到 0）
- **Clip Fraction**: 裁剪比例，应在 0.1-0.3 之间
- **Sharpe**: 验证集/测试集夏普比率

## 🎯 算法对比

### SAC vs PPO

| 特性 | SAC | PPO |
|------|-----|-----|
| 策略类型 | Off-policy | On-policy |
| 探索机制 | 最大熵 | 熵系数衰减 |
| 样本效率 | 高（replay buffer）| 低（每次 rollout 后清空）|
| 训练稳定性 | 中（Q-value 估计难）| 高（clip 限制）|
| 超参数敏感度 | 高 | 中 |
| 适合金融？ | ❌ 熵噪声大 | ✅ 可控制探索 |

## 🐛 故障排除

### 问题：训练初期 Sharpe 很低
- **正常**：PPO 需要更多 step 来稳定策略
- **解决**：增加 `total_timesteps_per_window` 到 200000

### 问题：Loss 不下降
- **检查**：奖励缩放是否正确（应放大 100x）
- **检查**：Advantage 是否归一化

### 问题：GPU 内存不足
- **解决**：减小 `batch_size` 到 1024
- **解决**：减小 `mini_batch_size` 到 32

### 问题：所有股票权重相同（均匀分布）
- **可能**：温度系数太高或太低
- **解决**：调整 `temperature` 在 0.5-2.0 之间

## 📁 交付物清单

| 文件 | 说明 |
|------|------|
| `config_ppo.py` | PPO 超参数配置 |
| `networks.py` | Actor-Critic 网络（支持 Action Masking）|
| `ppo_agent.py` | PPO 算法实现（GAE + PPO-Clip）|
| `trainer_ppo.py` | PPO 训练循环 |
| `main_ppo.py` | 训练入口 |
| `environment.py` | 环境（奖励缩放 + action_mask 输出）|

## 🎓 PPO 算法流程

```
For each window:
    For each update:
        1. Collect Rollout (2048 steps)
           - 使用当前策略收集轨迹
           - 存储 (s, a, r, v, log_prob, done, mask)
        
        2. Compute GAE
           - 计算 Advantage: A_t = delta_t + gamma*lambda*A_{t+1}
           - 计算 Returns: R_t = A_t + V(s_t)
           - 归一化 Advantage
        
        3. Update Network (10 epochs)
           - For each mini-batch (64 samples):
             - 计算新 log_prob 和 value
             - Policy Loss: -min(ratio*A, clip(ratio)*A)
             - Value Loss: MSE(value, returns)
             - Entropy Loss: -entropy * coef
             - Total Loss = Policy + 0.5*Value + Entropy
             - 反向传播 + 梯度裁剪
        
        4. Clear Buffer
           - On-policy: 清空 rollout buffer
```

## 📈 预期结果

- **Sharpe Ratio**: 目标 > 0.5（跑赢等权基准）
- **Max Drawdown**: 目标 < 20%
- **Turnover**: 目标 < 50%（控制交易成本）
- **HHI**: 目标 > 0.05（适度集中，非均匀分布）

---

**版本**: PPO v1.0  
**更新日期**: 2024
