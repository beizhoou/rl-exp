#!/bin/bash
# 快速实验脚本 - 测试3组最有希望的超参数
# 使用方法: bash quick_experiments.sh

cd /root/autodl-tmp/exper_rl

# 创建实验目录
mkdir -p experiments

# 备份原配置
cp config_ppo.py config_ppo.py.backup

echo "=================================="
echo "🔬 开始3组快速实验"
echo "每组: 2窗口 × 50000步"
echo "=================================="

# ========== 实验1: 低学习率 ==========
echo ""
echo "🧪 实验1/3: 低学习率 (lr=1e-4)"
sed -i 's/lr: float = 3e-4/lr: float = 1e-4/' config_ppo.py
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --total-timesteps 50000 --n-windows 2 2>&1 | tee experiments/exp1_low_lr.log
cp config_ppo.py.backup config_ppo.py

# ========== 实验2: 小Batch高频更新 ==========
echo ""
echo "🧪 实验2/3: 小Batch高频更新 (batch_size=1024)"
sed -i 's/batch_size: int = 2048/batch_size: int = 1024/' config_ppo.py
sed -i 's/mini_batch_size: int = 64/mini_batch_size: int = 32/' config_ppo.py
sed -i 's/n_rollout_steps: int = 2048/n_rollout_steps: int = 1024/' config_ppo.py
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --total-timesteps 50000 --n-windows 2 2>&1 | tee experiments/exp2_small_batch.log
cp config_ppo.py.backup config_ppo.py

# ========== 实验3: 组合优化 ==========
echo ""
echo "🧪 实验3/3: 组合优化 (lr=1e-4, batch=1024, entropy=0.03)"
sed -i 's/lr: float = 3e-4/lr: float = 1e-4/' config_ppo.py
sed -i 's/batch_size: int = 2048/batch_size: int = 1024/' config_ppo.py
sed -i 's/mini_batch_size: int = 64/mini_batch_size: int = 32/' config_ppo.py
sed -i 's/n_rollout_steps: int = 2048/n_rollout_steps: int = 1024/' config_ppo.py
sed -i 's/entropy_coef: float = 0.01/entropy_coef: float = 0.03/' config_ppo.py
sed -i 's/val_window_months: int = 1/val_window_months: int = 2/' config_ppo.py
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --total-timesteps 50000 --n-windows 2 2>&1 | tee experiments/exp3_combo.log
cp config_ppo.py.backup config_ppo.py

# 恢复原配置
cp config_ppo.py.backup config_ppo.py
rm config_ppo.py.backup

# 解析结果
echo ""
echo "=================================="
echo "📊 实验结果汇总"
echo "=================================="

grep -h "Val Sharpe:" experiments/*.log | tail -6

echo ""
echo "详细日志保存在 experiments/ 目录"
