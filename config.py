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
    
    # 特征维度配置（总计40维）
    # 1. 价量技术面特征 (15维)
    tech_features: List[str] = None  # 将在运行时填充
    
    # 2. 基本面特征 (8维)
    fundamental_features: List[str] = None
    
    # 3. 股吧情绪特征 (3维)
    guba_features: List[str] = field(default_factory=lambda: ["bullishness", "panic", "consensus"])
    
    # 4. VLM研报特征 (8维)
    vlm_features: List[str] = field(default_factory=lambda: [
        "sentiment_score", "rating_change", "eps_g_y0", 
        "eps_g_y1", "eps_g_y2", "profit_revision", 
        "revenue_revision", "pe_forward_y1"
    ])
    
    # 数据参数
    n_stocks: int = 471
    n_features: int = 40  # tech(15) + fundamental(8) + guba(3) + vlm(8) + derived(6)
    lookback_window: int = 20
    
    # 时间范围（实际数据范围：2019.01.01 - 2025.12.31）
    train_months: int = 12
    test_months: int = 1
    start_date: str = "2019-01-01"
    end_date: str = "2025-12-31"
    
    # 缺失值处理参数
    vlm_valid_days: int = 5      # 研报有效期（交易日）
    sentiment_valid_days: int = 3  # 情绪数据有效期
    fundamental_valid_days: int = 60  # 基本面数据有效期（季度）

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

@dataclass
class ModelConfig:
    """模型架构"""
    d_model: int = 128                    # 特征编码维度
    nhead: int = 4                        # 注意力头数
    dropout: float = 0.1
    num_layers: int = 2                   # Transformer层数
    use_fast_encoder: bool = True

@dataclass
class SACConfig:
    """SAC算法超参数"""
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2                    # 初始温度参数
    buffer_size: int = 100000
    batch_size: int = 256
    update_interval: int = 5              # 每5步更新一次
    use_amp: bool = True                  # 混合精度训练
    max_grad_norm: float = 1.0            # 梯度裁剪
    target_entropy: float = None          # 自动计算：-n_stocks * 0.5

@dataclass
class TrainingConfig:
    """训练流程配置"""
    n_splits: int = 12                    # 滚动窗口数量
    episodes_per_window: int = 5          # 每个窗口训练轮数
    warmup_steps: int = 1000              # 预热步数
    max_steps_per_episode: int = 10000    # 每轮最大步数
    device: str = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    n_workers: int = 4
    
    # 早停配置
    early_stop_patience: int = 3          # 早停耐心值
    min_sharpe_improvement: float = 0.05  # 最小Sharpe改善
    
    # 输出目录
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    plot_dir: str = "./plots"

# 初始化配置实例
data_cfg = DataConfig()
trading_cfg = TradingConfig()
model_cfg = ModelConfig()
sac_cfg = SACConfig()
train_cfg = TrainingConfig()

# 自动计算target entropy
if sac_cfg.target_entropy is None:
    sac_cfg.target_entropy = -data_cfg.n_stocks * 0.5

# 创建输出目录
for path in [train_cfg.log_dir, train_cfg.save_dir, train_cfg.plot_dir]:
    os.makedirs(path, exist_ok=True)

# 打印配置信息
def print_config():
    """打印当前配置"""
    print("="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"\nData Config:")
    print(f"  Stocks: {data_cfg.n_stocks}")
    print(f"  Features: {data_cfg.n_features}")
    print(f"  Lookback: {data_cfg.lookback_window}")
    print(f"  Date Range: {data_cfg.start_date} to {data_cfg.end_date}")
    print(f"\nTrading Config:")
    print(f"  Transaction Cost: {trading_cfg.transaction_cost:.4%}")
    print(f"  Max Position: {trading_cfg.max_position:.1%}")
    print(f"  T+1: {trading_cfg.t_plus_1}")
    print(f"  Long Only: {trading_cfg.long_only}")
    print(f"\nModel Config:")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  nhead: {model_cfg.nhead}")
    print(f"\nTraining Config:")
    print(f"  Device: {train_cfg.device}")
    print(f"  Episodes/Window: {train_cfg.episodes_per_window}")
    print(f"  Batch Size: {sac_cfg.batch_size}")
    print("="*60)

if __name__ == "__main__":
    print_config()
