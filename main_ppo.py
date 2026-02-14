#!/usr/bin/env python3
"""
PPO Training Entry Point
=======================

A 股投资组合管理 - PPO 算法训练入口

使用方法:
    # 基础训练（使用预处理数据）
    python main_ppo.py --preprocessed-data processed_data.pkl
    
    # 完整流程（含基准对比）
    python main_ppo.py --preprocessed-data processed_data.pkl --run-benchmarks
    
    # 从原始数据训练
    python main_ppo.py
"""

import pandas as pd
import numpy as np
import torch
import argparse
import warnings
import os

warnings.filterwarnings('ignore')

# PPO 配置
from config_ppo import data_cfg, trading_cfg, model_cfg, ppo_cfg, train_cfg, print_config


def load_data(args):
    """加载数据"""
    if args.preprocessed_data and os.path.exists(args.preprocessed_data):
        from data_loader import PreprocessedDataLoader
        loader = PreprocessedDataLoader(data_cfg)
        df = loader.load(args.preprocessed_data)
    elif args.preprocessed_data:
        raise FileNotFoundError(f"预处理文件不存在: {args.preprocessed_data}")
    else:
        from data_loader import MultiSourceDataLoader
        loader = MultiSourceDataLoader(data_cfg)
        df = loader.load_and_merge()
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='PPO for A-Share Portfolio Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用预处理数据训练（推荐）
    python main_ppo.py --preprocessed-data processed_data.pkl
    
    # 训练并对比基准策略
    python main_ppo.py --preprocessed-data processed_data.pkl --run-benchmarks
    
    # 从原始数据训练
    python main_ppo.py
    
    # 仅运行基准对比
    python main_ppo.py --preprocessed-data processed_data.pkl --benchmark-only
        """
    )
    
    # 数据参数
    parser.add_argument('--preprocessed-data', type=str, default=None,
                       help='从预处理文件加载数据 (.pkl)')
    
    # 训练参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='每窗口总训练步数')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='PPO batch size (rollout steps)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    
    # 基准对比
    parser.add_argument('--run-benchmarks', action='store_true',
                       help='训练后运行基准对比')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='仅运行基准对比')
    parser.add_argument('--max-windows', type=int, default=None,
                       help='最大训练窗口数（用于快速实验）')
    
    args = parser.parse_args()
    
    # 更新配置
    train_cfg.device = args.device if torch.cuda.is_available() else 'cpu'
    if args.total_timesteps:
        train_cfg.total_timesteps_per_window = args.total_timesteps
    if args.batch_size:
        ppo_cfg.batch_size = args.batch_size
    if args.lr:
        ppo_cfg.lr = args.lr
        ppo_cfg.critic_lr = args.lr
    if args.max_windows:
        train_cfg.max_windows = args.max_windows
    
    # 打印配置
    print_config()
    
    # 仅运行基准
    if args.benchmark_only:
        from run_benchmarks import main as benchmark_main
        benchmark_main()
        return
    
    # 加载数据
    print(f"\n{'='*70}")
    print("🚀 Data Loading")
    print(f"{'='*70}")
    df = load_data(args)
    
    # 训练
    print(f"\n{'='*70}")
    print("🎯 Training: PPO for Portfolio Management")
    print(f"{'='*70}")
    
    from trainer_ppo import PPOTrainer
    trainer = PPOTrainer(df, train_cfg, data_cfg, trading_cfg, model_cfg, ppo_cfg)
    results = trainer.run()
    
    print(f"\n{'='*70}")
    print("✅ PPO Training Completed!")
    print(f"{'='*70}")
    print(f"\n📁 Checkpoints: {train_cfg.save_dir}")
    print(f"📁 Logs: {train_cfg.log_dir}")
    print(f"📁 Plots: {train_cfg.plot_dir}")
    
    # 运行基准对比
    if args.run_benchmarks:
        print(f"\n{'='*70}")
        print("📊 Running Benchmark Comparison")
        print(f"{'='*70}")
        
        from benchmarks import run_all_benchmarks
        from run_benchmarks import compare_with_rl, save_results, plot_comparison
        
        benchmark_results = run_all_benchmarks(df, train_cfg, verbose=True)
        
        output_dir = './benchmark_results'
        save_results(benchmark_results, output_dir)
        plot_comparison(benchmark_results, output_dir)
        compare_with_rl(results, benchmark_results, output_dir)
        
        print(f"\n📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
