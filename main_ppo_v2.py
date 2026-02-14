"""
PPO V2 主程序 - 修复负奖励问题

使用方法:
1. 调试模式（推荐首次运行）:
   python main_ppo_v2.py --debug-zero-cost --enable-cash --reward-mode profit_only

2. 正常训练:
   python main_ppo_v2.py --enable-cash --reward-mode risk_adjusted

3. 课程学习:
   python main_ppo_v2.py --enable-cash --curriculum
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import torch

# V2 导入
from config_ppo_v2 import (
    data_cfg, trading_cfg, model_cfg, ppo_cfg, train_cfg, print_config
)
from trainer_ppo_v2 import PPOTrainerV2


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PPO V2 for A-Share Portfolio Management')
    
    # V2: 调试模式
    parser.add_argument('--debug-zero-cost', action='store_true',
                       help='调试模式：训练时使用0交易成本')
    parser.add_argument('--debug-mode', action='store_true',
                       help='启用详细日志输出')
    
    # V2: 现金仓位
    parser.add_argument('--enable-cash', action='store_true', default=True,
                       help='启用现金仓位（第n+1维资产）')
    parser.add_argument('--no-cash', action='store_true',
                       help='禁用现金仓位')
    
    # V2: 奖励模式
    parser.add_argument('--reward-mode', type=str, default='risk_adjusted',
                       choices=['profit_only', 'log_return', 'sharpe', 'risk_adjusted'],
                       help='奖励函数模式')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                       help='奖励缩放系数')
    parser.add_argument('--turnover-penalty', type=float, default=0.3,
                       help='换手率惩罚系数（建议0.1-0.5）')
    parser.add_argument('--risk-penalty', type=float, default=10.0,
                       help='风险（方差）惩罚系数')
    
    # V2: 课程学习
    parser.add_argument('--curriculum', action='store_true',
                       help='启用课程学习')
    
    # V2: 策略分布
    parser.add_argument('--policy-type', type=str, default='softmax_gaussian',
                       choices=['dirichlet', 'softmax_gaussian'],
                       help='策略分布类型')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-K截断（如10表示只保留前10只股票）')
    
    # 交易成本
    parser.add_argument('--transaction-cost', type=float, default=0.0015,
                       help='交易成本（双边）')
    
    # 训练控制
    parser.add_argument('--max-windows', type=int, default=None,
                       help='最大窗口数（用于快速测试）')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='PPO batch size')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='学习率')
    
    # 数据路径
    parser.add_argument('--data-path', type=str, default=None,
                       help='数据文件路径（默认为processed_data.csv）')
    
    return parser.parse_args()


def apply_args_to_config(args):
    """将命令行参数应用到配置"""
    
    # V2: 调试模式
    trading_cfg.debug_zero_cost = args.debug_zero_cost
    train_cfg.debug_mode = args.debug_mode
    
    # V2: 现金仓位
    trading_cfg.enable_cash = not args.no_cash if args.no_cash else args.enable_cash
    
    # V2: 奖励配置
    trading_cfg.reward_mode = args.reward_mode
    trading_cfg.reward_scale = args.reward_scale
    trading_cfg.turnover_penalty_coef = args.turnover_penalty
    trading_cfg.risk_penalty_coef = args.risk_penalty
    trading_cfg.transaction_cost = args.transaction_cost
    
    # V2: 课程学习
    if args.curriculum:
        trading_cfg.curriculum_stages = [
            {'reward_mode': 'profit_only', 'cost': 0.0, 'min_updates': 30},
            {'reward_mode': 'log_return', 'cost': 0.0, 'min_updates': 30},
            {'reward_mode': 'risk_adjusted', 'cost': args.transaction_cost, 'min_updates': 100},
        ]
    
    # V2: 策略分布
    model_cfg.policy_distribution = args.policy_type
    model_cfg.top_k = args.top_k
    
    # 训练控制
    train_cfg.max_windows = args.max_windows
    ppo_cfg.batch_size = args.batch_size
    ppo_cfg.mini_batch_size = args.batch_size
    ppo_cfg.lr = args.lr
    ppo_cfg.critic_lr = args.lr
    
    # 根据现金仓位调整资产数
    if trading_cfg.enable_cash:
        data_cfg.n_stocks = 471  # 保持原始股票数
    
    return args.data_path or 'processed_data.csv'


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    data_path = apply_args_to_config(args)
    
    # 打印配置
    print_config()
    
    # 加载数据
    print(f"\n📂 加载数据: {data_path}")
    try:
        df = pd.read_csv(data_path)
        print(f"   数据形状: {df.shape}")
        print(f"   列: {list(df.columns[:10])}...")
    except FileNotFoundError:
        print(f"❌ 错误: 找不到数据文件 {data_path}")
        print("   请先运行数据预处理脚本")
        return
    
    # 数据验证
    required_cols = ['date', 'share_code', 'close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"❌ 错误: 缺少必要的列: {missing_cols}")
        return
    
    # 检查特征列
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    print(f"   特征列数量: {len(feature_cols)}")
    
    # V2: 初始化训练器
    print("\n🚀 初始化PPO V2训练器...")
    trainer = PPOTrainerV2(
        df=df,
        train_config=train_cfg,
        data_config=data_cfg,
        trading_config=trading_cfg,
        model_config=model_cfg,
        ppo_config=ppo_cfg
    )
    
    # 运行训练
    print("\n" + "="*70)
    print("开始训练")
    print("="*70)
    results = trainer.run()
    
    # 保存结果
    if results:
        import json
        output_file = os.path.join(train_cfg.plot_dir, 'results_summary.json')
        with open(output_file, 'w') as f:
            # 转换numpy类型为Python类型
            serializable_results = []
            for r in results:
                sr = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                      for k, v in r.items() if k != 'metrics_history'}
                serializable_results.append(sr)
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 结果已保存到: {output_file}")
    
    print("\n✅ 训练完成!")


if __name__ == "__main__":
    main()
