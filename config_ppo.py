"""
PPO (Proximal Policy Optimization) 配置
========================================

金融强化学习基准配置
针对 A 股投资组合管理任务优化
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataConfig:
    """多源数据配置"""
    # 数据源路径
    guba_dir: str = "/root/autodl-tmp/data/guba_sentiment_results"      # 股吧情绪
    basic_dir: str = "/root/autodl-tmp/data/stcok_basic"               # 价量基本面  
    vlm_file: str = "/root/autodl-tmp/data/vlm_sentiment_analysis_qwen30_a3b_low_dpi.csv"  # VLM研报
    
    # 数据字段映射
    date_col: str = "date"
    stock_col: str = "share_code"  # 统一使用share_code作为股票标识
    
    # 特征维度配置（动态支持任意维度，不再硬性要求40维）
    # 1. 价量技术面特征
    tech_features: List[str] = None  # 将在运行时填充
    
    # 2. 基本面特征
    fundamental_features: List[str] = None
    
    # 3. 股吧情绪特征
    guba_features: List[str] = field(default_factory=lambda: ["bullishness", "panic", "consensus"])
    
    # 4. VLM研报特征
    vlm_features: List[str] = field(default_factory=lambda: [
        "sentiment_score", "rating_change", "eps_g_y0", 
        "eps_g_y1", "eps_g_y2", "profit_revision", 
        "revenue_revision", "pe_forward_y1"
    ])
    
    # 数据参数
    n_stocks: int = 471
    n_features: int = None  # 将在运行时自动检测（f_* 开头的列数）
    lookback_window: int = 20
    
    # 滚动窗口配置
    train_window_months: int = 24      # 训练窗口月份数
    val_window_months: int = 1         # 验证窗口月份数 (用于早停)
    test_window_months: int = 1        # 测试窗口月份数 (实盘模拟)
    rolling_step_months: int = 1       # 每次滚动步长
    
    # 时间范围
    start_date: str = "2019-01-01"
    end_date: str = "2025-12-31"
    
    # 缺失值处理参数
    vlm_valid_days: int = 5            # 研报有效期（交易日）
    sentiment_valid_days: int = 3      # 情绪数据有效期
    fundamental_valid_days: int = 60   # 基本面数据有效期（季度）

@dataclass
class TradingConfig:
    """A股交易约束配置"""
    transaction_cost: float = 0.0015      # 双边千1.5
    max_position: float = 0.10            # 单股上限10%
    risk_free_rate: float = 0.02/252      # 日度无风险利率（年化2%）
    
    # A股特有约束
    long_only: bool = True                # A股做空限制
    t_plus_1: bool = True                 # T+1交易制度
    limit_up_pct: float = 0.10            # 涨停幅度（非ST股票）
    limit_down_pct: float = 0.10          # 跌停幅度（非ST股票）
    st_limit_pct: float = 0.05            # ST股票涨跌停幅度
    
    # 交易控制
    use_tradable_mask: bool = True        # 使用可交易mask
    min_trade_unit: int = 100             # 最小交易单位（1手=100股）
    
    # PPO 奖励缩放
    reward_scale: float = 100.0           # 奖励缩放系数（关键！将收益率放大100倍）

@dataclass
class ModelConfig:
    """模型架构"""
    d_model: int = 128                    # 特征编码维度
    nhead: int = 4                        # 注意力头数
    dropout: float = 0.1
    num_layers: int = 2                   # Transformer层数
    
    # Actor 温度系数（防止梯度消失）
    temperature: float = 1.0              # Softmax 温度系数

@dataclass
class PPOConfig:
    """PPO 算法超参数 - 金融强化学习基准配置"""
    lr: float = 3e-4                      # 学习率 (Actor 和 Critic)
    critic_lr: float = 3e-4               # Critic 学习率（可单独设置）
    gamma: float = 0.99                   # 折扣因子，关注长期收益
    gae_lambda: float = 0.95              # GAE 系数，平衡方差和偏差
    
    # PPO Clip 参数
    clip_range: float = 0.2               # PPO 截断范围，防止策略突变
    clip_range_vf: float = None           # Value function clip（可选）
    
    # 熵正则化（关键！防止过早收敛）
    entropy_coef: float = 0.01            # 初始熵系数
    entropy_coef_decay: bool = True       # 是否衰减熵系数
    entropy_coef_final: float = 0.0       # 最终熵系数
    
    # Value Loss 系数
    value_coef: float = 1.0               # Value loss 权重（配合 returns 归一化）
    
    # 训练控制
    batch_size: int = 2048                # 每次收集的步数（减小防止OOM，原4096太大）
    mini_batch_size: int = 64             # 更新时的切片大小（减小防止OOM）
    n_epochs: int = 5                     # 每次收集后更新的轮数（减小，防止过拟合）
    
    # 梯度裁剪
    max_grad_norm: float = 0.5            # 梯度裁剪阈值
    
    # 优化器
    optimizer_eps: float = 1e-5           # Adam epsilon
    weight_decay: float = 0.0             # L2 正则化

@dataclass
class TrainingConfig:
    """训练流程配置"""
    # 窗口配置
    @property
    def train_window_months(self):
        return data_cfg.train_window_months
    
    @property
    def val_window_months(self):
        return data_cfg.val_window_months
    
    @property
    def test_window_months(self):
        return data_cfg.test_window_months
    
    @property
    def rolling_step_months(self):
        return data_cfg.rolling_step_months
    
    # 训练控制
    total_timesteps_per_window: int = 100000  # 每个窗口的总训练步数（约50次更新）
    n_rollout_steps: int = 2048               # 每次收集的轨迹长度（与batch_size匹配）
    
    # 验证和早停
    eval_interval: int = 4096                 # 评估间隔（每2轮评估一次）
    early_stop_patience: int = 10             # 早停耐心值（增加，允许更多探索）
    min_sharpe_improvement: float = 0.005     # 最小 Sharpe 改善（降低，更容易触发）
    
    # 增量学习配置
    inherit_weights: bool = True              # 是否继承上一窗口权重
    
    # 系统配置
    device: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    n_workers: int = 4
    seed: int = 42                            # 随机种子
    
    # 输出目录
    log_dir: str = "./logs_ppo"
    save_dir: str = "./checkpoints_ppo"
    plot_dir: str = "./plots_ppo"

# 初始化配置实例
data_cfg = DataConfig()
trading_cfg = TradingConfig()
model_cfg = ModelConfig()
ppo_cfg = PPOConfig()
train_cfg = TrainingConfig()

# 创建输出目录
for path in [train_cfg.log_dir, train_cfg.save_dir, train_cfg.plot_dir]:
    os.makedirs(path, exist_ok=True)

# 打印配置信息
def print_config():
    """打印当前配置"""
    print("="*70)
    print("Configuration Summary - PPO for A-Share Portfolio Management")
    print("="*70)
    print(f"\n📊 Data Config:")
    print(f"  Stocks: {data_cfg.n_stocks}")
    print(f"  Features: {data_cfg.n_features}")
    print(f"  Lookback: {data_cfg.lookback_window}")
    
    print(f"\n🎯 PPO Config:")
    print(f"  Learning Rate: {ppo_cfg.lr}")
    print(f"  Gamma: {ppo_cfg.gamma}")
    print(f"  GAE Lambda: {ppo_cfg.gae_lambda}")
    print(f"  Clip Range: {ppo_cfg.clip_range}")
    print(f"  Entropy Coef: {ppo_cfg.entropy_coef} -> {ppo_cfg.entropy_coef_final}")
    print(f"  Batch Size: {ppo_cfg.batch_size}")
    print(f"  Mini-batch Size: {ppo_cfg.mini_batch_size}")
    print(f"  Update Epochs: {ppo_cfg.n_epochs}")
    
    print(f"\n💰 Trading Config:")
    print(f"  Transaction Cost: {trading_cfg.transaction_cost:.4%}")
    print(f"  Max Position: {trading_cfg.max_position:.1%}")
    print(f"  Reward Scale: {trading_cfg.reward_scale}x (关键！)")
    print(f"  T+1: {trading_cfg.t_plus_1}")
    
    print(f"\n🧠 Model Config:")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  Temperature: {model_cfg.temperature}")
    
    print(f"\n⚙️  Training Config:")
    print(f"  Device: {train_cfg.device}")
    print(f"  Total Timesteps/Window: {train_cfg.total_timesteps_per_window:,}")
    print("="*70)

if __name__ == "__main__":
    print_config()
