"""
PPO (Proximal Policy Optimization) 配置 V2
============================================

修复负奖励问题的改进配置

核心改进:
1. 支持现金仓位 (n_stocks + 1)
2. 课程学习支持
3. 改进的奖励函数
4. 调试模式（0交易成本）
5. 更强的换手率惩罚
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class DataConfig:
    """多源数据配置"""
    # 数据源路径
    guba_dir: str = "/root/autodl-tmp/data/guba_sentiment_results"
    basic_dir: str = "/root/autodl-tmp/data/stcok_basic"
    vlm_file: str = "/root/autodl-tmp/data/vlm_sentiment_analysis_qwen30_a3b_low_dpi.csv"
    
    # 数据字段映射
    date_col: str = "date"
    stock_col: str = "share_code"
    
    # 特征维度配置
    tech_features: List[str] = None
    fundamental_features: List[str] = None
    guba_features: List[str] = field(default_factory=lambda: ["bullishness", "panic", "consensus"])
    vlm_features: List[str] = field(default_factory=lambda: [
        "sentiment_score", "rating_change", "eps_g_y0", 
        "eps_g_y1", "eps_g_y2", "profit_revision", 
        "revenue_revision", "pe_forward_y1"
    ])
    
    # 数据参数
    n_stocks: int = 471
    n_features: int = None  # 运行时自动检测
    lookback_window: int = 20
    
    # 滚动窗口配置
    train_window_months: int = 24
    val_window_months: int = 3
    test_window_months: int = 1
    rolling_step_months: int = 1
    
    # 时间范围
    start_date: str = "2019-01-01"
    end_date: str = "2025-12-31"
    
    # 缺失值处理参数
    vlm_valid_days: int = 5
    sentiment_valid_days: int = 3
    fundamental_valid_days: int = 60

@dataclass
class TradingConfig:
    """A股交易约束配置 V2"""
    
    # ========== V2: 交易成本配置 ==========
    # 双边交易成本（默认千1.5）
    transaction_cost: float = 0.0015
    
    # V2: 调试模式 - 训练时使用0交易成本
    # 建议：先用0成本训练，确认模型能学会低买高卖，再逐渐加回成本
    debug_zero_cost: bool = False  # 设为True进行无成本调试
    
    # V2: 课程学习 - 逐步增加交易成本
    # 阶段1: 0成本 -> 阶段2: 0.0005 -> 阶段3: 0.0015
    curriculum_cost_schedule: List[float] = field(default_factory=lambda: [0.0, 0.0005, 0.0015])
    
    # 仓位限制
    max_position: float = 0.10  # 单股上限10%
    risk_free_rate: float = 0.02/252
    
    # A股特有约束
    long_only: bool = True
    t_plus_1: bool = True
    limit_up_pct: float = 0.10
    limit_down_pct: float = 0.10
    st_limit_pct: float = 0.05
    
    # 交易控制
    use_tradable_mask: bool = True
    min_trade_unit: int = 100
    
    # ========== V2: 现金仓位支持 ==========
    enable_cash: bool = True  # 启用现金仓位（第472维）
    
    # ========== V2: 改进的奖励配置 ==========
    # 奖励模式选择：
    # - 'profit_only': 纯收益（课程学习第一阶段）
    # - 'log_return': 对数收益率
    # - 'sharpe': 夏普比率
    # - 'risk_adjusted': 收益 - 风险惩罚 - 换手惩罚（推荐）
    reward_mode: str = "risk_adjusted"
    
    reward_scale: float = 1.0  # V2: 降低缩放，避免数值爆炸
    
    # V2: 风险惩罚系数（方差惩罚）
    risk_penalty_coef: float = 10.0
    
    # V2: 换手率惩罚系数（显著增加以抑制高频交易）
    # 原值: 0.01, 建议: 0.1~0.5
    turnover_penalty_coef: float = 0.3
    
    # 集中度惩罚系数
    concentration_penalty: float = 0.1
    
    # ========== V2: 课程学习配置 ==========
    # 分阶段训练：
    # 阶段1: profit_only + 0成本 -> 学会基本交易
    # 阶段2: risk_adjusted + 0成本 -> 学会风险控制  
    # 阶段3: risk_adjusted + 正常成本 -> 学会控制换手
    curriculum_stages: List[dict] = field(default_factory=lambda: [
        {'reward_mode': 'profit_only', 'cost': 0.0, 'min_updates': 50},
        {'reward_mode': 'log_return', 'cost': 0.0, 'min_updates': 50},
        {'reward_mode': 'risk_adjusted', 'cost': 0.0015, 'min_updates': 100},
    ])

@dataclass
class ModelConfig:
    """模型架构 V2"""
    # V2: 注意n_stocks+1用于现金仓位
    d_model: int = 128
    nhead: int = 4
    dropout: float = 0.1
    num_layers: int = 2
    
    # Actor 温度系数
    temperature: float = 1.0
    
    # V2: 策略分布选择
    # - 'dirichlet': Dirichlet分布（默认）
    # - 'softmax_gaussian': Softmax + 高斯噪声（更稳定）
    policy_distribution: str = "dirichlet"

@dataclass
class PPOConfig:
    """PPO 算法超参数 V2"""
    lr: float = 0.0003
    critic_lr: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO Clip 参数
    clip_range: float = 0.2
    clip_range_vf: float = None
    
    # 熵正则化
    entropy_coef: float = 0.01
    entropy_coef_decay: bool = True
    entropy_coef_final: float = 0.0
    
    # Value Loss 系数
    value_coef: float = 1.0
    
    # V2: 减小batch size以稳定训练
    batch_size: int = 512  # 从2048减小
    mini_batch_size: int = 512
    n_epochs: int = 5
    
    # 梯度裁剪
    max_grad_norm: float = 0.5
    
    # 优化器
    optimizer_eps: float = 1e-5
    weight_decay: float = 0.0

@dataclass
class TrainingConfig:
    """训练流程配置"""
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
    total_timesteps_per_window: int = 100000
    n_rollout_steps: int = 512
    
    # 验证和早停
    eval_interval: int = 4096
    early_stop_patience: int = 10
    min_sharpe_improvement: float = 0.005
    
    # 增量学习配置
    inherit_weights: bool = True
    
    # 系统配置
    device: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    n_workers: int = 4
    seed: int = 42
    
    # 输出目录
    log_dir: str = "./logs_ppo_v2"
    save_dir: str = "./checkpoints_ppo_v2"
    plot_dir: str = "./plots_ppo_v2"
    
    # V2: 调试模式
    debug_mode: bool = False  # 启用详细日志
    max_windows: int = None  # 限制窗口数量用于快速测试

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
    print("Configuration Summary - PPO V2 for A-Share Portfolio Management")
    print("="*70)
    print(f"\n📊 Data Config:")
    print(f"  Stocks: {data_cfg.n_stocks}")
    print(f"  Features: {data_cfg.n_features}")
    print(f"  Lookback: {data_cfg.lookback_window}")
    
    print(f"\n💰 Trading Config V2:")
    print(f"  Transaction Cost: {trading_cfg.transaction_cost:.4%}")
    print(f"  Debug Zero Cost: {trading_cfg.debug_zero_cost}")
    print(f"  Enable Cash: {trading_cfg.enable_cash}")
    print(f"  Reward Mode: {trading_cfg.reward_mode}")
    print(f"  Risk Penalty Coef: {trading_cfg.risk_penalty_coef}")
    print(f"  Turnover Penalty Coef: {trading_cfg.turnover_penalty_coef}")
    print(f"  Max Position: {trading_cfg.max_position:.1%}")
    
    print(f"\n🎯 PPO Config:")
    print(f"  Learning Rate: {ppo_cfg.lr}")
    print(f"  Gamma: {ppo_cfg.gamma}")
    print(f"  GAE Lambda: {ppo_cfg.gae_lambda}")
    print(f"  Clip Range: {ppo_cfg.clip_range}")
    print(f"  Entropy Coef: {ppo_cfg.entropy_coef} -> {ppo_cfg.entropy_coef_final}")
    print(f"  Batch Size: {ppo_cfg.batch_size}")
    print(f"  Mini-batch Size: {ppo_cfg.mini_batch_size}")
    print(f"  Update Epochs: {ppo_cfg.n_epochs}")
    
    print(f"\n🧠 Model Config:")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  Temperature: {model_cfg.temperature}")
    print(f"  Policy Distribution: {model_cfg.policy_distribution}")
    
    print(f"\n⚙️  Training Config:")
    print(f"  Device: {train_cfg.device}")
    print(f"  Total Timesteps/Window: {train_cfg.total_timesteps_per_window:,}")
    print(f"  Debug Mode: {train_cfg.debug_mode}")
    print("="*70)

if __name__ == "__main__":
    print_config()
