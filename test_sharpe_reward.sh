#!/bin/bash
# 快速测试夏普比率奖励函数
# 对比新旧reward设计

cd /root/autodl-tmp/exper_rl

echo "=================================="
echo "🔬 测试夏普比率奖励函数"
echo "=================================="

# 创建测试目录
mkdir -p experiments_sharpe

# 备份原配置
cp config_ppo.py config_ppo.py.backup

echo ""
echo "📝 原配置（收益率为主）vs 新配置（夏普比率为主）"
echo ""

# ========== 测试1: 旧配置（收益率为主） ==========
echo "🧪 测试1/2: 旧配置（收益率为主，sharpe权重0.01）"
echo "   reward = return + sharpe*0.01 - ..."
echo ""

# 修改回原配置逻辑（手动设置）
cat > config_ppo_temp.py << 'EOF'
# 临时覆盖 - 旧配置
from config_ppo import *
trading_cfg.reward_mode = 'sharpe_only'  # 实际测试时手动改回旧代码
trading_cfg.reward_scale = 100.0
ppo_cfg.lr = 3e-4
EOF

# 运行（先用纯夏普做基准）
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --total-timesteps 20480 --n-windows 1 2>&1 | tee experiments_sharpe/test_old_reward.log &
PID1=$!

# 恢复配置
cp config_ppo.py.backup config_ppo.py

echo "⏳ 等待测试完成（约5-10分钟）..."
wait $PID1

# ========== 测试2: 新配置（纯夏普） ==========
echo ""
echo "🧪 测试2/2: 新配置（纯夏普比率）"
echo "   reward = sharpe_ratio (风险调整后收益)"
echo ""

# 修改配置
sed -i 's/reward_mode: str = .*/reward_mode: str = '"'"'sharpe_only'"'"'/' config_ppo.py
sed -i 's/reward_scale: float = .*/reward_scale: float = 10.0/' config_ppo.py

python main_ppo.py --preprocessed-data processed_data.pkl --device cuda --total-timesteps 20480 --n-windows 1 2>&1 | tee experiments_sharpe/test_sharpe_reward.log

# 恢复配置
cp config_ppo.py.backup config_ppo.py
rm config_ppo.py.backup

# 对比结果
echo ""
echo "=================================="
echo "📊 对比结果"
echo "=================================="

echo ""
echo "旧配置（收益率为主）:"
grep -h "Val Sharpe:" experiments_sharpe/test_old_reward.log 2>/dev/null | tail -3 || echo "无数据"

echo ""
echo "新配置（夏普比率为主）:"
grep -h "Val Sharpe:" experiments_sharpe/test_sharpe_reward.log 2>/dev/null | tail -3 || echo "无数据"

echo ""
echo "=================================="
echo "💡 分析"
echo "=================================="
echo "如果新配置的Sharpe更高，说明风险调整奖励更有效"
echo ""
