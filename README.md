# 多模态强化学习投资组合交易系统 (Multi-Modal RL Portfolio Trading)

基于 **PPO** (Proximal Policy Optimization) 和 **SAC** (Soft Actor-Critic) 算法的 A 股投资组合交易系统，采用**带验证集的滚动微调步进回测**方案，融合股吧情绪、基本面、VLM研报三类异构数据源。

> ⚠️ **重要提示**：当前推荐使用 **PPO 算法**（默认），相比 SAC 更稳定，解决了梯度消失和熵噪声问题。

## 🏆 核心特性：滚动微调步进回测 (Rolling Walk-Forward with Fine-tuning)

针对 A 股**非平稳（Non-stationary）、高噪声、风格切换快**的市场特点，本系统采用顶级量化对冲基金的标准范式：

### 窗口设计

```
┌─────────────────────────────────────────────────────────────────┐
│  训练集 (Train)          │ 验证集 (Val) │ 测试集 (Test)        │
│  24个月                  │ 1个月        │ 1个月                 │
│  用于梯度更新            │ 用于早停     │ 实盘模拟（记录净值）  │
└─────────────────────────────────────────────────────────────────┘
```

### 滚动示意图

| 轮次 (Step) | 训练集 (Train) | 验证集 (Val) | **测试集 (Test)** | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 1** | 2019.01 - 2020.12 | 2021.01 | **2021.02** | 初始冷启动训练 |
| **Step 2** | 2019.02 - 2021.01 | 2021.02 | **2021.03** | 继承权重，快速微调 |
| **Step 3** | 2019.03 - 2021.02 | 2021.03 | **2021.04** | 保持数据新鲜度 |
| ... | ... | ... | ... | ... |

### 关键优势

1. **验证集早停 (Early Stopping)**
   - RL训练极其不稳定，使用验证集选出**表现最稳健的Checkpoint**
   - 避免过拟合到训练集

2. **增量学习 (Incremental Learning)**
   - 每轮加载上一轮训练好的权重
   - PPO: 使用相同网络结构快速适应新数据
   - 解决灾难性遗忘，快速适应新行情

3. **严格防止未来函数**
   - 每个窗口独立进行特征标准化
   - Fit on Train, Transform on Val/Test
   - 杜绝数据泄露

4. **工程优化**
   - **奖励缩放**：将收益率放大 100 倍，解决梯度消失
   - **Action Masking**：停牌/跌停股票权重强制为 0
   - **Advantage 归一化**：稳定 PPO 训练

## 🚀 算法对比：PPO vs SAC

| 特性 | PPO (推荐) | SAC |
|------|-----------|-----|
| 策略类型 | On-policy | Off-policy |
| 稳定性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 训练速度 | 中等 | 较快 |
| 熵控制 | 线性衰减 0.01→0 | 固定温度 |
| 数值稳定性 | 高（Dirichlet分布） | 中（Softmax+log） |
| 适用场景 | 金融序列决策 | 连续控制 |

## 📁 项目结构

```
exper_rl/
├── README.md                   # 项目说明文档（本文件）
├── config.py                   # SAC 配置（遗留）
├── config_ppo.py               # PPO 配置 ⭐推荐⭐
├── data_loader.py              # 多源数据加载与40维特征工程
├── preprocess_data.py          # 数据预处理脚本
├── environment.py              # A股交易环境（奖励缩放+Action Mask）
├── networks.py                 # 特征编码器 + Actor/Critic（支持PPO/SAC）
├── ppo_agent.py                # PPO算法实现（Dirichlet+GAE）⭐核心⭐
├── trainer_ppo.py              # PPO滚动训练器 ⭐核心⭐
├── main_ppo.py                 # PPO训练入口 ⭐推荐⭐
├── agent.py                    # SAC算法智能体（遗留）
├── trainer.py                  # SAC滚动训练器（遗留）
├── main.py                     # SAC训练入口（遗留）
├── utils.py                    # 工具类（Buffer+标准化+可视化）
├── benchmarks.py               # 7个对比策略集合
├── run_benchmarks.py           # 基准策略回测脚本
├── PPO_USAGE.md                # PPO使用详细指南
└── example_usage.sh            # 使用示例脚本
```

## 🎯 快速开始（PPO - 推荐）

### 1. 安装依赖

```bash
pip install torch pandas numpy gym tqdm matplotlib seaborn tensorboard
```

### 2. 数据预处理（只需执行一次）

```bash
cd /root/autodl-tmp/exper_rl

# 基础预处理（输出 processed_data.pkl）
python preprocess_data.py

# 或指定输出路径和日期范围
python preprocess_data.py \
    -o ./data/processed_2020_2023.pkl \
    --start-date 2020-01-01 \
    --end-date 2023-12-31
```

### 3. PPO 训练

```bash
# 基础训练（使用预处理数据，秒级启动）
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda

# 训练并对比基准策略
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --run-benchmarks

# 自定义参数
python main_ppo.py \
    --preprocessed-data processed_data.pkl \
    --device cuda \
    --total-timesteps 200000 \
    --batch-size 2048 \
    --lr 3e-4
```

### 4. 仅运行对比策略

```bash
# 使用预处理数据
python main_ppo.py --preprocessed-data processed_data.pkl --benchmark-only

# 或从原始数据重新生成
python run_benchmarks.py
```

## ⚙️ PPO 关键超参数

在 `config_ppo.py` 中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE系数，平衡偏差和方差 |
| `clip_range` | 0.2 | PPO截断范围 |
| `entropy_coef` | 0.01→0 | 熵系数（线性衰减）|
| `batch_size` | 4096 | Rollout收集步数 |
| `mini_batch_size` | 128 | 更新时的切片大小 |
| `n_epochs` | 5 | 每次收集后的更新轮数 |
| `total_timesteps_per_window` | 200000 | 每窗口总训练步数 |
| `n_rollout_steps` | 4096 | 每次收集的轨迹长度 |
| `reward_scale` | 100.0 | **关键！**收益率放大倍数 |

## 📊 异构特征融合 (40维)

| 特征组 | 维度 | 说明 |
|--------|------|------|
| 技术面 | ~15 | price_position, returns, volatility, MACD, RSI, volume... |
| 基本面 | 8 | pe_ttm, pb, peg, ps, roe, growth, gross_margin, debt_ratio |
| 股吧情绪 | 3 | bullishness, panic, consensus |
| VLM研报 | 8 | sentiment_score, rating_change, eps_g_y0~y2, profit_revision... |
| 交互/派生 | ~6 | sentiment_consensus, momentum复合因子等 |

## 💰 A股交易约束

- **T+1制度**: 当日买入次日才能卖出
- **涨跌停限制**: 涨停不能买，跌停不能卖
- **停牌处理**: 自动检测并排除停牌股票
- **做空限制**: 只能做多
- **单股仓位上限**: 10%
- **交易成本**: 双边 0.15%

## 🏆 对比策略 (Benchmarks)

| 策略 | 逻辑 | 检验目标 |
|------|------|---------|
| **CSI300_Benchmark** | 沪深300指数收益 | 市场Beta基准 |
| **Equal_Weight** | 每日等权持有 | 消除选股，检验择时 |
| **Buy_Hold** | 期初买入持有不调仓 | 交易频率价值 |
| **Momentum_20D** | 做多20日收益率Top10% | 动量因子有效性 |
| **Mean_Reversion** | 做多20日收益率Bottom10% | 反转因子有效性 |
| **Low_Volatility** | 做多20日波动率最低10% | 低波动异象 |
| **Sentiment_Driven** | 做多情绪得分最高10% | 多模态情绪价值 |

## 📂 数据要求

```
/root/autodl-tmp/data/
├── guba_sentiment_results/     # 股吧情绪，471个 *_result.csv
├── stcok_basic/                # 基本面数据，471个 .csv
└── vlm_sentiment_analysis_*.csv # VLM研报分析结果，单个文件
```

## 📈 输出结果

```
./logs_ppo/              # TensorBoard 日志（PPO）
./checkpoints_ppo/       # 模型权重
./plots_ppo/             # 训练报告图
./benchmark_results/     # 对比策略结果
```

## 🔍 可视化监控

```bash
# PPO 训练日志
tensorboard --logdir=./logs_ppo

# SAC 训练日志（如使用）
tensorboard --logdir=./logs
```

## 🐛 故障排除

### 问题：训练出现 NaN/Inf

**原因**：
- Softmax + log 导致 log(0) = -inf
- 输入数据含有 NaN/Inf

**解决**：已修复！当前使用 Dirichlet 分布替代 Softmax，并添加数据清洗。

### 问题：Reward 恒定为负数

**原因**：
- 奖励信号太弱（0.01 级别）
- 网络无法学习

**解决**：已添加奖励缩放（100x），现在奖励在 [-5, +5] 范围。

### 问题：GPU 内存不足

```bash
# 减小 batch size
python main_ppo.py --preprocessed-data processed_data.pkl --batch-size 1024
```

### 问题：所有股票权重相同

**原因**：温度系数不合适

**解决**：调整 `config_ppo.py` 中的 `temperature` 参数（0.5-2.0）。

## 📋 更新日志

### 2025-02-14: PPO超参数优化与TensorBoard增强

#### 超参数调整（训练更稳定、收敛更快）
| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| `batch_size` | 2048 | **4096** | 适中平衡效率与稳定性 |
| `mini_batch_size` | 64 | **128** | 增大使梯度估计更稳定 |
| `n_epochs` | 10 | **5** | 减小防止过拟合训练数据 |
| `total_timesteps_per_window` | 100k | **200k** | 约100次更新，提高收敛 |
| `n_rollout_steps` | 2048 | **4096** | 匹配batch_size |
| `eval_interval` | 2048 | **8192** | 每2轮评估一次 |
| `early_stop_patience` | 5 | **10** | 允许更多探索 |
| `min_sharpe_improvement` | 0.01 | **0.005** | 更容易触发早停 |

#### TensorBoard增强
- **新增指标**：`Train/Reward` - 每回合平均奖励
- **完整指标列表**：
  - `Loss/Critic` - Value Loss
  - `Loss/Actor` - Policy Loss
  - `Train/Entropy` - 熵损失
  - `Train/Reward` - 平均回合奖励 ⭐新增
  - `Portfolio/*` - 组合价值、收益、Sharpe、换手率
  - `Eval/val/*`, `Eval/test/*` - 验证/测试集表现

#### 训练稳定性修复
- **Value Loss过大问题**：添加Returns归一化（均值0，方差1）
- **Returns范围**：从 ~[-1000, +1000] 归一化到 ~[-3, +3]

## 📝 待办/优化方向

- [ ] 特征重要性分析 (SHAP值)
- [ ] 分组注意力（按特征来源分头）
- [ ] 研报时效性加权（新研报权重更高）
- [ ] 行业中性约束
- [ ] 多进程数据加载加速
- [ ] 添加 Risk Parity / Minimum Variance 基准
- [ ] 自适应学习率（根据市场波动调整）
- [ ] 模型集成（多窗口模型投票）

## 📚 相关文档

- `PPO_USAGE.md` - PPO 算法详细使用指南
- `NA_FIX_SUMMARY.md` - NaN/Inf 问题修复总结
- `example_usage.sh` - 常用命令示例

---

**作者**: Multi-Modal RL Trading System  
**版本**: 1.0.0 (PPO + SAC Dual Algorithm Support)  
**更新日期**: 2024
