import pandas as pd
import numpy as np
import torch
import argparse
import warnings
import os

warnings.filterwarnings('ignore')

from config import data_cfg, trading_cfg, model_cfg, sac_cfg, train_cfg, print_config

class ConfigWrapper:
    """包装所有配置，便于trainer访问"""
    def __init__(self):
        self.data = data_cfg
        self.trading = trading_cfg
        self.model = model_cfg
        self.sac = sac_cfg
        self.train = train_cfg
        
config_wrapper = ConfigWrapper()
from data_loader import MultiSourceDataLoader
from trainer import RollingTrainer

def main():
    parser = argparse.ArgumentParser(description='Multi-Modal RL Portfolio Trading System')
    parser.add_argument('--windows', type=int, default=12, help='Number of rolling windows')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--episodes', type=int, default=None, help='Episodes per window')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--save-features', action='store_true', help='Save fused features to CSV')
    parser.add_argument('--run-benchmarks', action='store_true', 
                       help='Run benchmark comparison after training')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Only run benchmarks without RL training')
    args = parser.parse_args()
    
    # 更新配置
    train_cfg.device = args.device if torch.cuda.is_available() else 'cpu'
    if args.episodes is not None:
        train_cfg.episodes_per_window = args.episodes
    if args.batch_size is not None:
        sac_cfg.batch_size = args.batch_size
    
    # 打印配置
    print_config()
    
    # 仅运行基准策略
    if args.benchmark_only:
        from run_benchmarks import main as benchmark_main
        benchmark_main()
        return
    
    print(f"\n{'='*70}")
    print("Data Loading & Feature Engineering")
    print(f"{'='*70}")
    
    # 数据加载与特征工程
    loader = MultiSourceDataLoader(data_cfg)
    df = loader.load_and_merge()
    
    # 保存融合后的数据（可选）
    if args.save_features:
        output_path = 'fused_features.csv'
        df.to_csv(output_path, index=False)
        print(f"\nFused data saved to {output_path}")
    
    print(f"\n{'='*70}")
    print("Training")
    print(f"{'='*70}")
    
    # 启动训练
    trainer = RollingTrainer(df, train_cfg, data_cfg, trading_cfg, model_cfg, sac_cfg)
    results = trainer.run(n_splits=args.windows)
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")
    print(f"\nCheckpoints saved to: {train_cfg.save_dir}")
    print(f"Logs saved to: {train_cfg.log_dir}")
    print(f"Plots saved to: {train_cfg.plot_dir}")
    
    if train_cfg.device == 'cuda':
        print(f"\nTo view tensorboard logs, run:")
        print(f"  tensorboard --logdir={train_cfg.log_dir}")
    
    # 运行基准对比
    if args.run_benchmarks:
        print(f"\n{'='*70}")
        print("Running Benchmark Comparison")
        print(f"{'='*70}")
        
        from benchmarks import run_all_benchmarks
        from run_benchmarks import compare_with_rl, save_results, plot_comparison
        
        # 运行基准策略
        benchmark_results = run_all_benchmarks(df, train_cfg, verbose=True)
        
        # 保存结果
        output_dir = './benchmark_results'
        save_results(benchmark_results, output_dir)
        plot_comparison(benchmark_results, output_dir)
        
        # 对比RL与基准
        compare_with_rl(results, benchmark_results, output_dir)
        
        print(f"\nBenchmark results saved to: {output_dir}")

if __name__ == "__main__":
    main()
