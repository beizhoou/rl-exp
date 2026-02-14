"""
夏普比率聚焦超参数搜索
======================
基于风险调整后收益的PPO超参数优化

核心思想：Reward = Sharpe Ratio（而非原始收益率）

使用方法:
python sharpe_focused_search.py --n-windows 3 --timesteps 50000
"""

import os
import sys
import json
import subprocess
import pandas as pd
from datetime import datetime
import argparse

# 基于夏普比率的实验配置组
SHARPE_EXPERIMENTS = {
    # ========== 奖励函数对比 ==========
    "sharpe_pure": {
        "name": "纯夏普比率",
        "description": "只优化Sharpe，最稳健的风险调整收益",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    "sharpe_balanced": {
        "name": "夏普+收益平衡",
        "description": "60%夏普 + 40%收益，兼顾风险和回报",
        "reward_mode": "sharpe_return_balanced",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    "sortino": {
        "name": "Sortino比率",
        "description": "只惩罚下行风险，适合偏好多头策略",
        "reward_mode": "sortino",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    "calmar": {
        "name": "Calmar比率",
        "description": "年化收益/最大回撤，严格控制回撤",
        "reward_mode": "calmar",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    
    # ========== 学习率调优（基于纯夏普） ==========
    "sharpe_lr1e-4": {
        "name": "夏普 + 低学习率",
        "description": "lr=1e-4，更稳定的学习过程",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 1e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    "sharpe_lr5e-4": {
        "name": "夏普 + 高学习率",
        "description": "lr=5e-4，更快收敛但可能不稳定",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 5e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    
    # ========== 探索vs利用调优 ==========
    "sharpe_high_entropy": {
        "name": "夏普 + 高探索",
        "description": "entropy=0.05，增加策略探索",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.05,
        "entropy_coef_final": 0.01,
        "gae_lambda": 0.95,
    },
    "sharpe_low_entropy": {
        "name": "夏普 + 低探索",
        "description": "entropy=0.005，更快收敛",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.005,
        "gae_lambda": 0.95,
    },
    
    # ========== GAE调优 ==========
    "sharpe_long_gae": {
        "name": "夏普 + 长程GAE",
        "description": "gae_lambda=0.99，关注长期优势",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.99,
    },
    "sharpe_short_gae": {
        "name": "夏普 + 短程GAE",
        "description": "gae_lambda=0.9，关注短期优势",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 2048,
        "entropy_coef": 0.01,
        "gae_lambda": 0.9,
    },
    
    # ========== Batch Size调优 ==========
    "sharpe_small_batch": {
        "name": "夏普 + 小Batch高频",
        "description": "batch=1024，更新频率翻倍",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 1024,
        "n_rollout_steps": 1024,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    "sharpe_large_batch": {
        "name": "夏普 + 大Batch稳定",
        "description": "batch=4096（如果显存够），梯度更稳定",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 3e-4,
        "batch_size": 4096,
        "n_rollout_steps": 4096,
        "entropy_coef": 0.01,
        "gae_lambda": 0.95,
    },
    
    # ========== 组合优化（推荐） ==========
    "sharpe_optimal": {
        "name": "夏普最优组合",
        "description": "低学习率 + 适度探索 + 长程GAE",
        "reward_mode": "sharpe_only",
        "reward_scale": 10.0,
        "lr": 1e-4,
        "batch_size": 1024,
        "n_rollout_steps": 1024,
        "entropy_coef": 0.03,
        "entropy_coef_final": 0.005,
        "gae_lambda": 0.97,
    },
}


def modify_config(exp_config, config_path="config_ppo.py"):
    """修改配置文件"""
    with open(config_path, "r") as f:
        content = f.read()
    
    import re
    
    # 修改 reward_mode
    if "reward_mode" in exp_config:
        pattern = r'(reward_mode:\s*str\s*=\s*).*?(\s+#|$)'
        replacement = rf'\g<1>"{exp_config["reward_mode"]}"\2'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 修改 PPO 参数
    ppo_mappings = {
        "lr": "lr: float",
        "batch_size": "batch_size: int",
        "entropy_coef": "entropy_coef: float",
        "entropy_coef_final": "entropy_coef_final: float",
        "gae_lambda": "gae_lambda: float",
        "n_rollout_steps": "n_rollout_steps: int",
    }
    
    for key, pattern_prefix in ppo_mappings.items():
        if key in exp_config:
            value = exp_config[key]
            if isinstance(value, str):
                value_str = f'"{value}"'
            else:
                value_str = str(value)
            
            pattern = rf'({pattern_prefix}\s*=\s*).*?(\s+#|$)'
            replacement = rf'\g<1>{value_str}\2'
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    with open(config_path, "w") as f:
        f.write(content)


def run_experiment(exp_name, exp_config, args):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"🧪 {exp_config['name']}")
    print(f"   {exp_config['description']}")
    print(f"{'='*70}")
    
    # 备份原配置
    os.system("cp config_ppo.py config_ppo.py.backup")
    
    # 修改配置
    modify_config(exp_config)
    
    # 创建实验目录
    exp_dir = f"experiments_sharpe/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(exp_config, f, indent=2, default=str)
    
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
        print(f"⏰ 实验超时")
        success = False
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        success = False
    
    # 恢复原配置
    os.system("cp config_ppo.py.backup config_ppo.py")
    
    # 解析结果
    results = parse_results(exp_name, exp_dir, log_file)
    
    return success, results


def parse_results(exp_name, exp_dir, log_file):
    """解析训练结果"""
    results = {
        "exp_name": exp_name,
        "final_val_sharpe": 0.0,
        "final_test_sharpe": 0.0,
        "avg_reward": 0.0,
        "best_val_sharpe": -999,
        "converged": False,
    }
    
    if not os.path.exists(log_file):
        return results
    
    with open(log_file, "r") as f:
        lines = f.readlines()
    
    rewards = []
    val_sharpes = []
    test_sharpes = []
    
    for line in lines:
        if "Reward=" in line:
            try:
                reward = float(line.split("Reward=")[1].split(",")[0])
                rewards.append(reward)
            except:
                pass
        
        if "Val Sharpe:" in line:
            try:
                sharpe = float(line.split("Val Sharpe:")[1].strip())
                val_sharpes.append(sharpe)
            except:
                pass
        
        if "Test Sharpe:" in line:
            try:
                sharpe = float(line.split("Test Sharpe:")[1].strip())
                test_sharpes.append(sharpe)
            except:
                pass
    
    if rewards:
        results["avg_reward"] = sum(rewards) / len(rewards)
        # 判断是否收敛（最后10个reward波动<1）
        if len(rewards) >= 20:
            recent_std = pd.Series(rewards[-10:]).std()
            results["converged"] = recent_std < 1.0 if not pd.isna(recent_std) else False
    
    if val_sharpes:
        results["final_val_sharpe"] = val_sharpes[-1]
        results["best_val_sharpe"] = max(val_sharpes)
    
    if test_sharpes:
        results["final_test_sharpe"] = test_sharpes[-1]
    
    return results


def print_summary(all_results):
    """打印结果汇总"""
    print(f"\n{'='*90}")
    print("📊 夏普聚焦实验结果汇总")
    print(f"{'='*90}")
    
    df_data = []
    for exp_name, results in all_results.items():
        config = SHARPE_EXPERIMENTS.get(exp_name, {})
        df_data.append({
            "实验": exp_name,
            "名称": config.get('name', ''),
            "奖励模式": config.get('reward_mode', ''),
            "学习率": config.get('lr', ''),
            "Best Val Sharpe": f"{results['best_val_sharpe']:.3f}",
            "Final Val Sharpe": f"{results['final_val_sharpe']:.3f}",
            "Final Test Sharpe": f"{results['final_test_sharpe']:.3f}",
            "收敛": "✓" if results['converged'] else "✗",
        })
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # 找出最佳
    best_exp = max(all_results.items(), key=lambda x: x[1]['best_val_sharpe'])
    print(f"\n🏆 最佳实验: {best_exp[0]}")
    print(f"   Best Val Sharpe: {best_exp[1]['best_val_sharpe']:.3f}")
    print(f"   配置: {json.dumps(SHARPE_EXPERIMENTS[best_exp[0]], indent=2, default=str)}")
    
    # 保存
    os.makedirs("experiments_sharpe", exist_ok=True)
    df.to_csv("experiments_sharpe/summary.csv", index=False)
    with open("experiments_sharpe/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 结果保存到 experiments_sharpe/")
    
    return best_exp[0]


def main():
    parser = argparse.ArgumentParser(description="夏普聚焦超参数搜索")
    parser.add_argument("--data-path", type=str, default="processed_data.pkl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-windows", type=int, default=2)
    parser.add_argument("--experiments", type=str, default="key",
                       help="key=关键5组, all=全部12组, 或逗号分隔")
    parser.add_argument("--timeout", type=int, default=3600)
    
    args = parser.parse_args()
    
    # 选择实验
    if args.experiments == "key":
        # 最关键的5组
        experiments_to_run = [
            "sharpe_pure",           # 基准
            "sharpe_balanced",       # 收益平衡
            "sharpe_lr1e-4",         # 低学习率
            "sharpe_high_entropy",   # 高探索
            "sharpe_optimal",        # 组合优化
        ]
    elif args.experiments == "all":
        experiments_to_run = list(SHARPE_EXPERIMENTS.keys())
    else:
        experiments_to_run = args.experiments.split(",")
    
    print(f"🔬 夏普聚焦超参数搜索")
    print(f"实验数量: {len(experiments_to_run)}")
    print(f"每实验: {args.n_windows}窗口 × {args.timesteps}步")
    print(f"核心思想: Reward = Sharpe Ratio (风险调整后收益)")
    
    # 运行实验
    all_results = {}
    for exp_name in experiments_to_run:
        if exp_name not in SHARPE_EXPERIMENTS:
            print(f"⚠️ 未知实验: {exp_name}")
            continue
        
        exp_config = SHARPE_EXPERIMENTS[exp_name]
        success, results = run_experiment(exp_name, exp_config, args)
        all_results[exp_name] = results
    
    # 汇总
    best = print_summary(all_results)
    
    print(f"\n📝 建议:")
    print(f"   使用配置: {best}")
    print(f"   修改 config_ppo.py 中的 reward_mode 和相应参数")


if __name__ == "__main__":
    main()
