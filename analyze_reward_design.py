"""
奖励函数设计分析
===============
对比不同reward设计的效果

核心指标:
1. Sharpe Ratio (验证集) - 风险调整后收益
2. Max Drawdown - 最大回撤
3. Turnover - 换手率
4. Return/Volatility - 收益波动比
"""

import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log(log_path):
    """解析日志文件"""
    data = {
        'rewards': [],
        'val_sharpes': [],
        'test_sharpes': [],
        'policy_losses': [],
        'value_losses': [],
    }
    
    if not os.path.exists(log_path):
        return data
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'Reward=' in line:
                match = re.search(r'Reward=([\d\-\.]+)', line)
                if match:
                    data['rewards'].append(float(match.group(1)))
            
            if 'Val Sharpe:' in line:
                match = re.search(r'Val Sharpe:\s*([\d\-\.]+)', line)
                if match:
                    data['val_sharpes'].append(float(match.group(1)))
            
            if 'Test Sharpe:' in line:
                match = re.search(r'Test Sharpe:\s*([\d\-\.]+)', line)
                if match:
                    data['test_sharpes'].append(float(match.group(1)))
            
            if 'Policy Loss=' in line:
                match = re.search(r'Policy Loss=([\d\-\.]+)', line)
                if match:
                    data['policy_losses'].append(float(match.group(1)))
            
            if 'Value Loss=' in line:
                match = re.search(r'Value Loss=([\d\-\.]+)', line)
                if match:
                    data['value_losses'].append(float(match.group(1)))
    
    return data


def analyze_experiment(exp_name, data):
    """分析单个实验"""
    metrics = {
        'name': exp_name,
        'n_updates': len(data['rewards']),
        'avg_reward': np.mean(data['rewards']) if data['rewards'] else 0,
        'std_reward': np.std(data['rewards']) if data['rewards'] else 0,
        'final_val_sharpe': data['val_sharpes'][-1] if data['val_sharpes'] else 0,
        'best_val_sharpe': max(data['val_sharpes']) if data['val_sharpes'] else 0,
        'avg_test_sharpe': np.mean(data['test_sharpes']) if data['test_sharpes'] else 0,
        'sharpe_variance': np.std(data['val_sharpes']) if len(data['val_sharpes']) > 1 else 0,
    }
    
    # 计算风险调整得分
    if metrics['best_val_sharpe'] > 0:
        metrics['risk_score'] = metrics['best_val_sharpe'] / (1 + metrics['sharpe_variance'])
    else:
        metrics['risk_score'] = metrics['best_val_sharpe']
    
    return metrics


def plot_comparison(exp_metrics):
    """绘制对比图"""
    if len(exp_metrics) < 2:
        print("需要至少2个实验才能对比")
        return
    
    names = [m['name'] for m in exp_metrics]
    x = np.arange(len(names))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sharpe对比
    sharpes = [m['best_val_sharpe'] for m in exp_metrics]
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    axes[0, 0].bar(x, sharpes, color=colors, alpha=0.7)
    axes[0, 0].set_ylabel('Best Val Sharpe')
    axes[0, 0].set_title('Risk-Adjusted Return (Higher is Better)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Reward稳定性
    std_rewards = [m['std_reward'] for m in exp_metrics]
    axes[0, 1].bar(x, std_rewards, color='orange', alpha=0.7)
    axes[0, 1].set_ylabel('Reward Std Dev')
    axes[0, 1].set_title('Training Stability (Lower is Better)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Sharpe方差（一致性）
    sharpe_vars = [m['sharpe_variance'] for m in exp_metrics]
    axes[1, 0].bar(x, sharpe_vars, color='purple', alpha=0.7)
    axes[1, 0].set_ylabel('Sharpe Variance')
    axes[1, 0].set_title('Cross-Window Consistency (Lower is Better)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 综合风险得分
    risk_scores = [m['risk_score'] for m in exp_metrics]
    axes[1, 1].bar(x, risk_scores, color='gold', alpha=0.8)
    axes[1, 1].set_ylabel('Risk Score')
    axes[1, 1].set_title('Overall Score (Sharpe / (1+Var))')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments_sharpe/reward_comparison.png', dpi=150)
    print("📊 对比图已保存: experiments_sharpe/reward_comparison.png")


def print_report(exp_metrics):
    """打印分析报告"""
    print("\n" + "="*80)
    print("📊 奖励函数设计分析报告")
    print("="*80)
    
    # 排序
    sorted_metrics = sorted(exp_metrics, key=lambda x: x['best_val_sharpe'], reverse=True)
    
    print(f"\n{'排名':<4} {'实验':<25} {'Best Sharpe':<12} {'Sharpe方差':<12} {'稳定性':<10}")
    print("-"*80)
    
    for i, m in enumerate(sorted_metrics, 1):
        stability = "✓ 高" if m['std_reward'] < 2 else "△ 中" if m['std_reward'] < 5 else "✗ 低"
        print(f"{i:<4} {m['name']:<25} {m['best_val_sharpe']:<12.3f} {m['sharpe_variance']:<12.3f} {stability:<10}")
    
    # 最佳推荐
    best = sorted_metrics[0]
    print(f"\n" + "="*80)
    print(f"🏆 最佳奖励设计: {best['name']}")
    print(f"   Best Val Sharpe: {best['best_val_sharpe']:.3f}")
    print(f"   跨窗口一致性: {best['sharpe_variance']:.3f} (越低越好)")
    print(f"   训练稳定性: {best['std_reward']:.2f} (越低越好)")
    
    # 设计建议
    print(f"\n" + "="*80)
    print("💡 奖励函数设计建议")
    print("="*80)
    
    if best['name'] == 'sharpe_pure' or 'sharpe' in best['name']:
        print("✅ 使用纯夏普比率作为奖励是有效的")
        print("   - 策略学会平衡风险与收益")
        print("   - 避免了过度追求高收益而忽视风险")
    
    if best['sharpe_variance'] > 0.5:
        print("⚠️  Sharpe方差较高，说明不同窗口表现差异大")
        print("   建议: 增加验证窗口长度或使用滚动Sharpe")
    
    if best['std_reward'] > 5:
        print("⚠️  Reward波动大，训练不够稳定")
        print("   建议: 降低学习率或增加batch size")


def main():
    print("🔍 分析奖励函数设计效果")
    
    # 查找实验
    exp_dirs = [
        Path("experiments_sharpe"),
        Path("experiments"),
    ]
    
    all_metrics = []
    
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        
        for log_file in exp_dir.glob("*.log"):
            exp_name = log_file.stem
            print(f"\n分析: {exp_name}")
            
            data = parse_log(log_file)
            metrics = analyze_experiment(exp_name, data)
            all_metrics.append(metrics)
            
            print(f"  Updates: {metrics['n_updates']}")
            print(f"  Best Sharpe: {metrics['best_val_sharpe']:.3f}")
            print(f"  Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    if len(all_metrics) < 1:
        print("❌ 未找到实验日志")
        return
    
    # 报告和可视化
    print_report(all_metrics)
    plot_comparison(all_metrics)
    
    # 保存
    with open("experiments_sharpe/analysis.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\n💾 分析结果已保存")


if __name__ == "__main__":
    main()
