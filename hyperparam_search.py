"""
PPO超参数搜索脚本
================
运行多组实验，找到最优配置

使用方法:
python hyperparam_search.py --n-windows 3 --timesteps 50000
"""

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime
from dataclasses import asdict
import argparse

# 实验配置组
EXPERIMENTS = {
    "baseline": {
        "name": "基线配置",
        "lr": 3e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 5,
        "entropy_coef": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "low_lr": {
        "name": "低学习率",
        "lr": 1e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 5,
        "entropy_coef": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "high_entropy": {
        "name": "高熵探索",
        "lr": 3e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 5,
        "entropy_coef": 0.05,  # 增加探索
        "entropy_coef_final": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "small_batch": {
        "name": "小Batch高频更新",
        "lr": 3e-4,
        "batch_size": 1024,  # 减半
        "mini_batch_size": 32,
        "n_epochs": 5,
        "entropy_coef": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "long_val": {
        "name": "长验证期",
        "lr": 3e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 5,
        "entropy_coef": 0.01,
        "val_window_months": 3,  # 3个月验证期
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "aggressive_clip": {
        "name": "激进裁剪",
        "lr": 3e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 10,  # 更多epoch
        "entropy_coef": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.95,
        "clip_range": 0.1,  # 更保守的裁剪
    },
    "tuned_gae": {
        "name": "调整GAE",
        "lr": 3e-4,
        "batch_size": 2048,
        "mini_batch_size": 64,
        "n_epochs": 5,
        "entropy_coef": 0.01,
        "val_window_months": 1,
        "gae_lambda": 0.99,  # 更长的优势估计
        "clip_range": 0.2,
    },
    "combo": {
        "name": "组合优化",
        "lr": 1e-4,
        "batch_size": 1024,
        "mini_batch_size": 32,
        "n_epochs": 5,
        "entropy_coef": 0.03,
        "entropy_coef_final": 0.005,
        "val_window_months": 2,
        "gae_lambda": 0.97,
        "clip_range": 0.15,
    },
}


def modify_config(exp_config, config_path="config_ppo.py"):
    """修改配置文件"""
    with open(config_path, "r") as f:
        content = f.read()
    
    # 替换配置
    for key, value in exp_config.items():
        if key in ["name"]:
            continue
        # 找到对应的行并替换
        import re
        pattern = rf"({key}:\s*).*?(\s+#|$)"
        replacement = rf"\g<1>{value}\2"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 保存到临时文件
    with open(config_path, "w") as f:
        f.write(content)


def run_experiment(exp_name, exp_config, args):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"🧪 开始实验: {exp_name} - {exp_config['name']}")
    print(f"{'='*70}")
    print(f"配置: {json.dumps(exp_config, indent=2, default=str)}")
    
    # 备份原配置
    os.system("cp config_ppo.py config_ppo.py.backup")
    
    # 修改配置
    modify_config(exp_config)
    
    # 创建实验目录
    exp_dir = f"experiments/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 运行训练
    cmd = [
        "python", "main_ppo.py",
        "--preprocessed-data", args.data_path,
        "--device", args.device,
        "--total-timesteps", str(args.timesteps),
        "--max-windows", str(args.n_windows),
    ]
    
    log_file = f"{exp_dir}/train.log"
    
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                cwd="/root/autodl-tmp/exper_rl"
            )
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验 {exp_name} 超时")
        success = False
    except Exception as e:
        print(f"❌ 实验 {exp_name} 失败: {e}")
        success = False
    
    # 恢复原配置
    os.system("cp config_ppo.py.backup config_ppo.py")
    
    # 解析结果
    results = parse_results(exp_name, exp_dir, log_file)
    
    return success, results


def parse_results(exp_name, exp_dir, log_file):
    """从日志解析训练结果"""
    results = {
        "exp_name": exp_name,
        "final_val_sharpe": 0.0,
        "final_test_sharpe": 0.0,
        "avg_reward": 0.0,
        "convergence_step": -1,
        "best_window_sharpe": -999,
    }
    
    if not os.path.exists(log_file):
        return results
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    rewards = []
    val_sharpes = []
    test_sharpes = []
    
    for line in lines:
        # 解析Reward
        if "Reward=" in line:
            try:
                reward = float(line.split("Reward=")[1].split(",")[0])
                rewards.append(reward)
            except:
                pass
        
        # 解析Val Sharpe
        if "Val Sharpe:" in line:
            try:
                sharpe = float(line.split("Val Sharpe:")[1].strip())
                val_sharpes.append(sharpe)
            except:
                pass
        
        # 解析Test Sharpe
        if "Test Sharpe:" in line:
            try:
                sharpe = float(line.split("Test Sharpe:")[1].strip())
                test_sharpes.append(sharpe)
            except:
                pass
    
    if rewards:
        results["avg_reward"] = sum(rewards) / len(rewards)
        # 找Reward开始稳定的位置
        for i in range(10, len(rewards)):
            recent = rewards[i-10:i]
            if max(recent) - min(recent) < 2.0:  # 波动小于2
                results["convergence_step"] = i
                break
    
    if val_sharpes:
        results["final_val_sharpe"] = val_sharpes[-1]
        results["best_window_sharpe"] = max(val_sharpes)
    
    if test_sharpes:
        results["final_test_sharpe"] = test_sharpes[-1]
    
    return results


def print_summary(all_results):
    """打印实验结果汇总"""
    print(f"\n{'='*80}")
    print("📊 实验结果汇总")
    print(f"{'='*80}")
    
    # 创建DataFrame
    df_data = []
    for exp_name, results in all_results.items():
        df_data.append({
            "实验": exp_name,
            "名称": EXPERIMENTS[exp_name]["name"],
            "最终Val Sharpe": f"{results['final_val_sharpe']:.3f}",
            "最佳Val Sharpe": f"{results['best_window_sharpe']:.3f}",
            "最终Test Sharpe": f"{results['final_test_sharpe']:.3f}",
            "平均Reward": f"{results['avg_reward']:.2f}",
            "收敛步数": results['convergence_step'],
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # 找出最佳配置
    best_exp = max(all_results.items(), key=lambda x: x[1]['best_window_sharpe'])
    print(f"\n🏆 最佳配置: {best_exp[0]} - {EXPERIMENTS[best_exp[0]]['name']}")
    print(f"   最佳Val Sharpe: {best_exp[1]['best_window_sharpe']:.3f}")
    
    # 保存结果
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    df.to_csv("experiments/results.csv", index=False)
    print(f"\n💾 结果已保存到 experiments/results.csv")


def main():
    parser = argparse.ArgumentParser(description="PPO超参数搜索")
    parser.add_argument("--data-path", type=str, default="processed_data.pkl",
                       help="预处理数据路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="训练设备")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="每窗口训练步数（默认50k快速测试）")
    parser.add_argument("--n-windows", type=int, default=2,
                       help="测试窗口数（默认2个，减少时间）")
    parser.add_argument("--experiments", type=str, default="all",
                       help="运行的实验，逗号分隔，默认all")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="单个实验超时时间（秒）")
    
    args = parser.parse_args()
    
    # 选择实验
    if args.experiments == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = args.experiments.split(",")
    
    print(f"🔬 将运行 {len(experiments_to_run)} 组实验")
    print(f"实验列表: {', '.join(experiments_to_run)}")
    print(f"每实验: {args.n_windows}窗口 × {args.timesteps}步")
    
    # 运行实验
    all_results = {}
    for exp_name in experiments_to_run:
        if exp_name not in EXPERIMENTS:
            print(f"⚠️ 未知实验: {exp_name}，跳过")
            continue
        
        exp_config = EXPERIMENTS[exp_name]
        success, results = run_experiment(exp_name, exp_config, args)
        all_results[exp_name] = results
        
        if success:
            print(f"✅ {exp_name} 完成")
        else:
            print(f"⚠️ {exp_name} 可能失败，但继续")
    
    # 汇总结果
    print_summary(all_results)


if __name__ == "__main__":
    main()
