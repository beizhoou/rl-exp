"""
对比策略回测运行脚本

使用方法:
    python run_benchmarks.py --data fused_features.csv --output benchmark_results.csv
    
或结合主训练流程:
    python main.py --run-benchmarks
"""

import pandas as pd
import numpy as np
import argparse
import os
import json
from datetime import datetime

from config import data_cfg, trading_cfg, model_cfg, sac_cfg, train_cfg
from data_loader import MultiSourceDataLoader
from benchmarks import run_all_benchmarks


def save_results(results: dict, output_dir: str = './benchmark_results'):
    """保存回测结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存汇总表格
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Strategy': name,
            'Total_Return_%': result['total_return'],
            'Annual_Return_%': result['annual_return'],
            'Sharpe': result['sharpe'],
            'Max_Drawdown_%': result['max_drawdown'],
            'Volatility_%': result['volatility'],
            'Win_Rate_%': result['win_rate'],
            'Avg_Turnover_%': result['avg_turnover']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'benchmark_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # 2. 保存净值曲线
    for name, result in results.items():
        nav_df = pd.DataFrame({
            'date': result.get('dates', range(len(result['history']))),
            'nav': result['history']
        })
        nav_path = os.path.join(output_dir, f'nav_{name}.csv')
        nav_df.to_csv(nav_path, index=False)
    
    # 3. 保存详细指标
    detailed_results = {}
    for name, result in results.items():
        detailed_results[name] = {
            'total_return': result['total_return'],
            'annual_return': result['annual_return'],
            'sharpe': result['sharpe'],
            'max_drawdown': result['max_drawdown'],
            'volatility': result['volatility'],
            'win_rate': result['win_rate'],
            'avg_turnover': result['avg_turnover']
        }
    
    json_path = os.path.join(output_dir, 'benchmark_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"Metrics saved to: {json_path}")
    
    return summary_df


def plot_comparison(results: dict, output_dir: str = './benchmark_results'):
    """绘制对比图表"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 净值曲线对比
        ax = axes[0, 0]
        for name, result in results.items():
            history = result['history']
            ax.plot(history, label=name, linewidth=1.5)
        ax.set_title('Portfolio Value Comparison', fontsize=14)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 2. 收益对比柱状图
        ax = axes[0, 1]
        names = list(results.keys())
        returns = [results[n]['total_return'] for n in names]
        colors = ['steelblue' if r > 0 else 'coral' for r in returns]
        bars = ax.bar(range(len(names)), returns, color=colors, alpha=0.7)
        ax.set_title('Total Return Comparison', fontsize=14)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Total Return (%)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, returns)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
        
        # 3. Sharpe vs Max Drawdown 散点图
        ax = axes[1, 0]
        sharpes = [results[n]['sharpe'] for n in names]
        drawdowns = [results[n]['max_drawdown'] for n in names]
        scatter = ax.scatter(sharpes, drawdowns, s=100, alpha=0.6, c=range(len(names)), cmap='viridis')
        for i, name in enumerate(names):
            ax.annotate(name, (sharpes[i], drawdowns[i]), fontsize=8, alpha=0.8)
        ax.set_title('Sharpe vs Max Drawdown', fontsize=14)
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Max Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # 4. 换手率对比
        ax = axes[1, 1]
        turnovers = [results[n]['avg_turnover'] for n in names]
        bars = ax.bar(range(len(names)), turnovers, color='teal', alpha=0.6)
        ax.set_title('Average Turnover Comparison', fontsize=14)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Avg Turnover (%)')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'benchmark_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {plot_path}")
        
    except ImportError:
        print("Warning: matplotlib/seaborn not found, skipping plot generation")


def compare_with_rl(rl_results: list, benchmark_results: dict, output_dir: str = './benchmark_results'):
    """
    对比RL策略与基准策略
    
    Args:
        rl_results: RL策略的滚动窗口结果列表
        benchmark_results: 基准策略结果字典
    """
    print("\n" + "="*80)
    print("RL vs Benchmarks Comparison")
    print("="*80)
    
    # 计算RL策略的平均表现
    if rl_results and len(rl_results) > 0:
        rl_sharpes = [r['sharpe'] for r in rl_results]
        rl_returns = [r['total_return_pct'] for r in rl_results]
        rl_drawdowns = [r['max_drawdown'] for r in rl_results]
        
        rl_avg_sharpe = np.mean(rl_sharpes)
        rl_avg_return = np.mean(rl_returns)
        rl_avg_dd = np.mean(rl_drawdowns)
        
        print(f"\nRL Strategy (Avg across {len(rl_results)} windows):")
        print(f"  Sharpe: {rl_avg_sharpe:.2f}")
        print(f"  Return: {rl_avg_return:.2f}%")
        print(f"  Max DD: {rl_avg_dd:.2%}")
        
        # 对比表格
        print(f"\n{'Strategy':<25} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10} {'vs_RL_Sharpe':>12}")
        print("-"*80)
        
        for name, result in benchmark_results.items():
            sharpe = result['sharpe']
            ret = result['total_return']
            dd = result['max_drawdown']
            diff = sharpe - rl_avg_sharpe
            diff_str = f"{diff:+.2f}"
            
            print(f"{name:<25} {sharpe:>10.2f} {ret:>11.2f}% {dd:>9.2f}% {diff_str:>12}")
        
        # 统计优于RL的基准
        better_benchmarks = [name for name, result in benchmark_results.items() 
                            if result['sharpe'] > rl_avg_sharpe]
        
        print(f"\nBenchmarks with Sharpe > RL: {len(better_benchmarks)}/{len(benchmark_results)}")
        if better_benchmarks:
            print(f"  {', '.join(better_benchmarks)}")
    else:
        print("No RL results provided for comparison")


def main():
    parser = argparse.ArgumentParser(description='Run Benchmark Strategies')
    parser.add_argument('--data', type=str, default=None, 
                       help='Path to fused_features.csv (if None, will run data loading)')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate comparison plots')
    args = parser.parse_args()
    
    print("="*80)
    print("Multi-Modal RL Portfolio Trading - Benchmark Comparison")
    print("="*80)
    
    # 加载数据
    if args.data and os.path.exists(args.data):
        print(f"\nLoading data from: {args.data}")
        df = pd.read_csv(args.data)
        df[data_cfg.date_col] = pd.to_datetime(df[data_cfg.date_col])
    else:
        print("\nRunning data loading pipeline...")
        loader = MultiSourceDataLoader(data_cfg)
        df = loader.load_and_merge()
        
        # 保存供后续使用
        fused_path = 'fused_features.csv'
        df.to_csv(fused_path, index=False)
        print(f"Data saved to: {fused_path}")
    
    print(f"\nDataset: {len(df)} rows, {df[data_cfg.stock_col].nunique()} stocks")
    print(f"Date range: {df[data_cfg.date_col].min()} to {df[data_cfg.date_col].max()}")
    
    # 运行基准策略
    results = run_all_benchmarks(df, train_cfg, verbose=True)
    
    # 保存结果
    summary_df = save_results(results, args.output)
    
    # 绘制图表
    if args.plot:
        plot_comparison(results, args.output)
    
    print("\n" + "="*80)
    print("Benchmark analysis completed!")
    print(f"Results saved to: {args.output}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    main()
