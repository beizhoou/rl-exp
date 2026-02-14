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


def load_data(args):
    """
    加载数据（支持从预处理文件或原始数据）
    """
    if args.preprocessed_data and os.path.exists(args.preprocessed_data):
        # 从预处理文件加载
        from data_loader import PreprocessedDataLoader
        loader = PreprocessedDataLoader(data_cfg)
        df = loader.load(args.preprocessed_data)
    elif args.preprocessed_data:
        print(f"❌ 预处理文件不存在: {args.preprocessed_data}")
        print(f"   请运行: python preprocess_data.py -o {args.preprocessed_data}")
        raise FileNotFoundError(f"预处理文件不存在: {args.preprocessed_data}")
    else:
        # 从原始数据加载（完整流程）
        from data_loader import MultiSourceDataLoader
        loader = MultiSourceDataLoader(data_cfg)
        df = loader.load_and_merge()
        
        # 可选：保存预处理后的数据
        if args.save_preprocessed:
            from preprocess_data import save_processed_data
            feature_cols = [c for c in df.columns if c.startswith('f_')]
            feature_stats = {
                'mean': df[feature_cols].mean(),
                'std': df[feature_cols].std() + 1e-8,
            }
            metadata = {
                'n_rows': len(df),
                'n_stocks': df[data_cfg.stock_col].nunique(),
                'n_features': len(feature_cols),
            }
            save_processed_data(df, feature_cols, feature_stats, 
                               args.save_preprocessed, metadata)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal RL Portfolio Trading System - Rolling Walk-Forward with Fine-tuning'
    )
    
    # 数据参数
    parser.add_argument('--preprocessed-data', type=str, default=None,
                       help='从预处理文件加载数据 (.pkl)，大幅提高启动速度')
    parser.add_argument('--save-preprocessed', type=str, default=None,
                       help='预处理后的数据保存路径（用于下次快速加载）')
    parser.add_argument('--save-features', action='store_true',
                       help='保存融合特征到CSV（兼容旧版本）')
    
    # 训练参数
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--episodes-first', type=int, default=None, 
                       help='Episodes for first window (cold start)')
    parser.add_argument('--episodes-finetune', type=int, default=None, 
                       help='Episodes for subsequent windows (fine-tuning)')
    parser.add_argument('--batch-size', type=int, default=None, 
                       help='Batch size for training')
    parser.add_argument('--no-inherit', action='store_true',
                       help='Disable weight inheritance (no incremental learning)')
    
    # 基准对比参数
    parser.add_argument('--run-benchmarks', action='store_true', 
                       help='Run benchmark comparison after training')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Only run benchmarks without RL training')
    
    # 预处理参数
    parser.add_argument('--preprocess-only', action='store_true',
                       help='仅执行数据预处理，保存后退出')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD)，覆盖配置')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD)，覆盖配置')
    
    args = parser.parse_args()
    
    # 仅执行预处理模式
    if args.preprocess_only:
        from preprocess_data import preprocess_data
        output_path = args.save_preprocessed or 'processed_data.pkl'
        preprocess_data(
            output_path=output_path,
            start_date=args.start_date,
            end_date=args.end_date
        )
        print(f"\n✅ 预处理完成，数据已保存到: {output_path}")
        print(f"下次训练可使用: python main.py --preprocessed-data {output_path}")
        return
    
    # 更新配置
    train_cfg.device = args.device if torch.cuda.is_available() else 'cpu'
    if args.episodes_first is not None:
        train_cfg.episodes_first_window = args.episodes_first
    if args.episodes_finetune is not None:
        train_cfg.episodes_finetune = args.episodes_finetune
    if args.batch_size is not None:
        sac_cfg.batch_size = args.batch_size
    if args.no_inherit:
        train_cfg.inherit_weights = False
    if args.start_date:
        data_cfg.start_date = args.start_date
    if args.end_date:
        data_cfg.end_date = args.end_date
    
    # 打印配置
    print_config()
    
    # 仅运行基准策略
    if args.benchmark_only:
        from run_benchmarks import main as benchmark_main
        benchmark_main()
        return
    
    print(f"\n{'='*70}")
    print("🚀 Data Loading")
    print(f"{'='*70}")
    
    # 加载数据
    df = load_data(args)
    
    # 保存CSV版本（可选，用于外部分析）
    if args.save_features:
        output_path = 'fused_features.csv'
        df.to_csv(output_path, index=False)
        print(f"\n💾 Fused data saved to {output_path}")
    
    print(f"\n{'='*70}")
    print("🎯 Training: Rolling Walk-Forward with Fine-tuning")
    print(f"{'='*70}")
    
    # 启动训练
    from trainer import RollingWalkForwardTrainer
    trainer = RollingWalkForwardTrainer(
        df, train_cfg, data_cfg, trading_cfg, model_cfg, sac_cfg
    )
    results = trainer.run()
    
    print(f"\n{'='*70}")
    print("✅ Training completed!")
    print(f"{'='*70}")
    print(f"\n📁 Checkpoints saved to: {train_cfg.save_dir}")
    print(f"📁 Logs saved to: {train_cfg.log_dir}")
    print(f"📁 Plots saved to: {train_cfg.plot_dir}")
    
    if train_cfg.device == 'cuda':
        print(f"\n📊 To view tensorboard logs, run:")
        print(f"   tensorboard --logdir={train_cfg.log_dir}")
    
    # 运行基准对比
    if args.run_benchmarks:
        print(f"\n{'='*70}")
        print("📊 Running Benchmark Comparison")
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
        
        print(f"\n📁 Benchmark results saved to: {output_dir}")


if __name__ == "__main__":
    main()
