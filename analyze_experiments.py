"""
实验结果分析脚本
==============
分析超参数实验结果，给出最优配置建议

使用方法:
python analyze_experiments.py
"""

import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """解析单个实验日志"""
    results = {
        "rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "val_sharpes": [],
        "test_sharpes": [],
    }
    
    if not os.path.exists(log_path):
        return results
    
    with open(log_path, "r") as f:
        for line in f:
            # Reward
            if "Reward=" in line and "Update" in line:
                match = re.search(r"Reward=([\d\-\.]+)", line)
                if match:
                    results["rewards"].append(float(match.group(1)))
            
            # Policy Loss
            if "Policy Loss=" in line:
                match = re.search(r"Policy Loss=([\d\-\.]+)", line)
                if match:
                    results["policy_losses"].append(float(match.group(1)))
            
            # Value Loss
            if "Value Loss=" in line:
                match = re.search(r"Value Loss=([\d\-\.]+)", line)
                if match:
                    results["value_losses"].append(float(match.group(1)))
            
            # Entropy
            if "Entropy=" in line and "Update" in line:
                match = re.search(r"Entropy=([\d\-\.]+)", line)
                if match:
                    results["entropies"].append(float(match.group(1)))
            
            # Val Sharpe
            if "Val Sharpe:" in line:
                match = re.search(r"Val Sharpe:\s*([\d\-\.]+)", line)
                if match:
                    results["val_sharpes"].append(float(match.group(1)))
            
            # Test Sharpe
            if "Test Sharpe:" in line:
                match = re.search(r"Test Sharpe:\s*([\d\-\.]+)", line)
                if match:
                    results["test_sharpes"].append(float(match.group(1)))
    
    return results


def analyze_experiment(exp_name, results):
    """分析单个实验"""
    metrics = {
        "name": exp_name,
        "n_updates": len(results["rewards"]),
        "avg_reward": np.mean(results["rewards"]) if results["rewards"] else 0,
        "final_reward": results["rewards"][-1] if results["rewards"] else 0,
        "reward_trend": "up" if len(results["rewards"]) > 10 and results["rewards"][-1] > results["rewards"][10] else "down/stable",
        "avg_policy_loss": np.mean(results["policy_losses"]) if results["policy_losses"] else 0,
        "final_value_loss": results["value_losses"][-1] if results["value_losses"] else 999,
        "final_entropy": results["entropies"][-1] if results["entropies"] else 0,
        "best_val_sharpe": max(results["val_sharpes"]) if results["val_sharpes"] else 0,
        "final_val_sharpe": results["val_sharpes"][-1] if results["val_sharpes"] else 0,
        "avg_test_sharpe": np.mean(results["test_sharpes"]) if results["test_sharpes"] else 0,
    }
    
    # 计算收敛性得分
    if len(results["rewards"]) > 20:
        early = np.mean(results["rewards"][:10])
        late = np.mean(results["rewards"][-10:])
        metrics["convergence"] = "yes" if abs(late - early) < 3 else "no"
    else:
        metrics["convergence"] = "unknown"
    
    return metrics


def plot_comparison(all_metrics):
    """绘制实验对比图"""
    if len(all_metrics) < 2:
        print("至少需要2个实验才能对比")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    exp_names = [m["name"] for m in all_metrics]
    x_pos = np.arange(len(exp_names))
    
    # 1. Val Sharpe对比
    val_sharpes = [m["best_val_sharpe"] for m in all_metrics]
    axes[0, 0].bar(x_pos, val_sharpes, color='steelblue')
    axes[0, 0].set_ylabel("Best Val Sharpe")
    axes[0, 0].set_title("Best Validation Sharpe")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Test Sharpe对比
    test_sharpes = [m["avg_test_sharpe"] for m in all_metrics]
    axes[0, 1].bar(x_pos, test_sharpes, color='green', alpha=0.7)
    axes[0, 1].set_ylabel("Avg Test Sharpe")
    axes[0, 1].set_title("Average Test Sharpe")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Final Value Loss
    value_losses = [m["final_value_loss"] for m in all_metrics]
    axes[0, 2].bar(x_pos, value_losses, color='orange', alpha=0.7)
    axes[0, 2].set_ylabel("Final Value Loss")
    axes[0, 2].set_title("Critic Learning (Lower Better)")
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Reward趋势
    rewards = [m["final_reward"] for m in all_metrics]
    axes[1, 0].bar(x_pos, rewards, color='purple', alpha=0.7)
    axes[1, 0].set_ylabel("Final Reward")
    axes[1, 0].set_title("Final Reward (Higher Better)")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Convergence
    conv_scores = [1 if m["convergence"] == "yes" else 0 for m in all_metrics]
    colors = ['green' if c == 1 else 'red' for c in conv_scores]
    axes[1, 1].bar(x_pos, conv_scores, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel("Converged (1=Yes)")
    axes[1, 1].set_title("Training Convergence")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1, 1].set_ylim(-0.1, 1.2)
    
    # 6. 综合得分
    # 综合得分 = Val Sharpe * 0.4 + Test Sharpe * 0.4 + (1/ValueLoss) * 0.2
    scores = []
    for m in all_metrics:
        val_score = max(0, m["best_val_sharpe"])  # 负Sharpe算0分
        test_score = max(0, m["avg_test_sharpe"])
        value_score = 1.0 / (1.0 + m["final_value_loss"])  # 归一化
        total = val_score * 0.4 + test_score * 0.4 + value_score * 0.2
        scores.append(total)
    
    axes[1, 2].bar(x_pos, scores, color='gold', alpha=0.8)
    axes[1, 2].set_ylabel("Composite Score")
    axes[1, 2].set_title("Overall Score (Higher Better)")
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("experiments/comparison.png", dpi=150)
    print("📊 对比图已保存: experiments/comparison.png")


def recommend_config(all_metrics):
    """推荐最优配置"""
    print("\n" + "="*70)
    print("🏆 最优配置推荐")
    print("="*70)
    
    # 按Best Val Sharpe排序
    sorted_metrics = sorted(all_metrics, key=lambda x: x["best_val_sharpe"], reverse=True)
    
    best = sorted_metrics[0]
    print(f"\n第一名: {best['name']}")
    print(f"  Best Val Sharpe: {best['best_val_sharpe']:.3f}")
    print(f"  Avg Test Sharpe: {best['avg_test_sharpe']:.3f}")
    print(f"  Final Value Loss: {best['final_value_loss']:.4f}")
    print(f"  Convergence: {best['convergence']}")
    
    if len(sorted_metrics) > 1:
        second = sorted_metrics[1]
        print(f"\n第二名: {second['name']}")
        print(f"  Best Val Sharpe: {second['best_val_sharpe']:.3f}")
    
    # 给出配置建议
    print("\n" + "="*70)
    print("📝 配置建议")
    print("="*70)
    
    if best["best_val_sharpe"] < 0.1:
        print("⚠️ 所有实验Sharpe都较低 (<0.1)，可能原因：")
        print("  1. 特征预测能力不足")
        print("  2. 市场环境不适合（2021年A股震荡）")
        print("  3. 需要更长的训练时间")
        print("  4. 可能需要调整奖励函数")
    else:
        print(f"✅ 推荐配置: {best['name']}")
        print(f"   可以达到 Sharpe = {best['best_val_sharpe']:.3f}")
    
    return best["name"]


def main():
    print("📊 PPO超参数实验结果分析")
    print("="*70)
    
    # 查找所有实验日志
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        print("❌ experiments/ 目录不存在，请先运行实验")
        return
    
    log_files = list(exp_dir.glob("*.log"))
    if not log_files:
        print("❌ 未找到实验日志文件 (*.log)")
        return
    
    print(f"找到 {len(log_files)} 个实验日志")
    
    # 分析每个实验
    all_metrics = []
    for log_file in log_files:
        exp_name = log_file.stem
        print(f"\n分析: {exp_name}")
        
        results = parse_log_file(log_file)
        metrics = analyze_experiment(exp_name, results)
        all_metrics.append(metrics)
        
        print(f"  Updates: {metrics['n_updates']}")
        print(f"  Best Val Sharpe: {metrics['best_val_sharpe']:.3f}")
        print(f"  Avg Test Sharpe: {metrics['avg_test_sharpe']:.3f}")
        print(f"  Convergence: {metrics['convergence']}")
    
    # 绘制对比图
    plot_comparison(all_metrics)
    
    # 推荐配置
    best_config = recommend_config(all_metrics)
    
    # 保存分析结果
    with open("experiments/analysis.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n💾 分析结果已保存: experiments/analysis.json")


if __name__ == "__main__":
    main()
