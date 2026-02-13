# 多模态强化学习投资组合交易系统 (Multi-Modal RL Portfolio Trading)

## 项目概述

基于 Soft Actor-Critic (SAC) 算法的 A 股投资组合交易系统，融合三类异构数据源：
- **股吧情绪数据** (Guba Sentiment): 散户情绪指标 (bullishness, panic, consensus)
- **基本面/价量数据** (Market Data): OHLCV、技术指标、财务指标
- **VLM 研报分析** (VLM Reports): 多模态大模型对研报图像的解析结果

## 项目结构

```
exper_rl/
├── README.md              # 项目说明文档
├── broker_weights.py      # 券商影响力权重配置
├── benchmarks.py          # 对比策略集合（7个基准策略）
├── run_benchmarks.py      # 基准策略回测运行脚本
├── config.py              # 全局配置
├── data_loader.py         # 多源数据加载与40维特征工程
├── environment.py         # A股交易环境 (含T+1、涨跌停约束)
├── agent.py               # SAC算法智能体
├── networks.py            # 异构特征编码器 + Actor/Critic网络
├── trainer.py             # 滚动训练器 (12月训练/1月测试)
├── utils.py               # 回放缓冲区、日志、可视化工具
└── main.py                # 训练入口
```

## 核心特性

### 1. 异构特征融合 (40维)
| 特征组 | 维度 | 说明 |
|--------|------|------|
| 技术面 | ~15 | price_position, returns, volatility, MACD, RSI, volume... |
| 基本面 | 8 | pe_ttm, pb, peg, ps, roe, growth, gross_margin, debt_ratio |
| 股吧情绪 | 3 | bullishness, panic, consensus |
| VLM研报 | 8 | sentiment_score, rating_change, eps_g_y0~y2, profit_revision... |
| 交互/派生 | ~6 | sentiment_consensus, momentum复合因子等 |

### 2. 券商影响力加权
VLM研报按券商影响力加权聚合：
- **Tier 1** (中信、中金等): 权重 1.0
- **Tier 2** (中大型): 权重 0.8
- **Tier 3** (中小): 权重 0.6
- **外资/其他**: 权重 0.5

### 3. A股交易约束
- **T+1制度**: 当日买入次日才能卖出
- **涨跌停限制**: 涨停不能买，跌停不能卖
- **停牌处理**: 自动检测并排除停牌股票
- **做空限制**: 只能做多
- **单股仓位上限**: 10%
- **交易成本**: 双边 0.15%

### 4. 训练流程
- **滚动窗口训练**: 12个月训练 → 1个月测试，共12个窗口
- **早停机制**: Sharpe无改善时提前停止
- **混合精度训练**: AMP加速
- **量化回放缓冲区**: uint8存储，节省75%内存

## 快速开始

### 安装依赖
```bash
pip install torch pandas numpy gym tqdm matplotlib seaborn tensorboard
```

### 训练RL策略
```bash
cd /root/autodl-tmp/exper_rl

# 基础训练
python main.py --windows 12 --device cuda

# 训练后自动运行对比策略
python main.py --windows 12 --run-benchmarks

# 保存融合特征（调试用）
python main.py --windows 12 --save-features --run-benchmarks
```

### 仅运行对比策略
```bash
# 使用已保存的特征数据
python run_benchmarks.py --data fused_features.csv

# 或从原始数据重新生成
python run_benchmarks.py

# 仅运行基准对比（不训练RL）
python main.py --benchmark-only
```

## 对比策略 (Benchmarks)

为验证RL策略有效性，实现了 **7个对比策略**，严格公平对比：

| 策略 | 逻辑 | 检验目标 |
|------|------|---------|
| **CSI300_Benchmark** | 沪深300指数收益 | 市场Beta基准 |
| **Equal_Weight** | 每日等权持有 | 消除选股，检验择时 |
| **Buy_Hold** | 期初买入持有不调仓 | 交易频率价值 |
| **Momentum_20D** | 做多20日收益率Top10% | 动量因子有效性 |
| **Mean_Reversion** | 做多20日收益率Bottom10% | 反转因子有效性 |
| **Low_Volatility** | 做多20日波动率最低10% | 低波动异象 |
| **Sentiment_Driven** | 做多情绪得分最高10% | 多模态情绪价值 |

**公平对比原则**:
- 相同股票池: 471只A股
- 相同交易成本: 0.15% 双边
- 相同约束: 单股10%上限，无做空
- 相同频率: 日度再平衡

## 数据要求

项目期望以下数据路径（可在 `config.py` 中修改）：

```
/root/autodl-tmp/data/
├── guba_sentiment_results/     # 股吧情绪，471个 *_result.csv
├── stcok_basic/                # 基本面数据，471个 .csv
└── vlm_sentiment_analysis_*.csv # VLM研报分析结果，单个文件
```

### 各数据源字段要求

**股吧情绪 (guba_sentiment_results/)**
```csv
share_code, date, bullishness, panic, consensus, summary, prompt, status
```

**基本面数据 (stcok_basic/)**
```csv
code/date, open, close, high, low, volume, daily_return, volatility_20, 
pe_ttm, pb, peg, ps, market_cap, roe, growth, gross_margin, debt_ratio
```

**VLM研报分析**
```csv
share_code, date, broker, sentiment_score, rating_change, eps_g_y0, eps_g_y1, eps_g_y2,
pe_forward_y1, profit_revision, revenue_revision, report_type, has_financial_table
```

## 输出结果

训练完成后会生成以下目录：
```
./logs/              # TensorBoard 日志
./checkpoints/       # 模型权重 (window{i}_best.pt, window{i}_final.pt)
./plots/             # 训练报告图
./benchmark_results/ # 对比策略结果
├── benchmark_summary.csv      # 汇总表格
├── benchmark_metrics.json     # 详细指标
├── nav_*.csv                  # 各策略净值曲线
└── benchmark_comparison.png   # 对比图表
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_stocks | 471 | 股票池数量 |
| n_features | 40 | 特征维度 |
| lookback_window | 20 | 时序回看天数 |
| d_model | 128 | Transformer隐藏维度 |
| transaction_cost | 0.0015 | 交易成本 |
| max_position | 0.10 | 单股仓位上限 |
| episodes_per_window | 5 | 每窗口训练轮数 |

## 可视化监控

```bash
tensorboard --logdir=./logs
```

可监控：
- 组合净值 / 日收益率 / Sharpe 比率
- Critic/Actor 损失
- 持仓权重分布 / HHI 集中度
- 各滚动窗口的测试表现

## 结果解读

### RL策略有效性判断
1. **Sharpe > CSI300**: 跑赢市场基准
2. **Sharpe > Equal_Weight**: 具备选股/择时能力
3. **Sharpe > Buy_Hold**: 主动管理有价值
4. **Sharpe > Sentiment_Driven**: 多模态融合优于单一情绪策略

### 可能的发现
- **数据优势**: RL整合多源数据，可能优于单一因子策略
- **动态适应**: 滚动训练使策略适应市场状态变化
- **成本控制**: RL学习低换手率策略，降低交易成本

## 待办/优化方向

- [ ] 特征重要性分析 (SHAP值)
- [ ] 分组注意力（按特征来源分头）
- [ ] 研报时效性加权（新研报权重更高）
- [ ] 行业中性约束
- [ ] 多进程数据加载加速
- [ ] 添加 Risk Parity / Minimum Variance 基准

---

## 使用示例

### 完整流程
```bash
# 1. 进入目录
cd /root/autodl-tmp/exper_rl

# 2. 训练RL策略并对比基准
python main.py --windows 12 --run-benchmarks

# 3. 查看结果
cat benchmark_results/benchmark_summary.csv
tensorboard --logdir=./logs
```

### 仅对比基准策略
```bash
python main.py --benchmark-only
```

### 自定义训练
```bash
python main.py --windows 12 --episodes 10 --batch-size 512 --run-benchmarks
```

---

**作者**: Multi-Modal RL Trading System  
**版本**: 0.2.0
