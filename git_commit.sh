#!/bin/bash
# Git 提交脚本 - PPO 算法升级

set -e

echo "======================================"
echo "Git 提交准备 - PPO 算法升级"
echo "======================================"
echo ""

# 1. 添加.gitignore
echo "1. 添加 .gitignore..."
git add .gitignore

# 2. 添加新增文件
echo "2. 添加 PPO 核心文件..."
git add config_ppo.py
git add ppo_agent.py
git add trainer_ppo.py
git add main_ppo.py

echo "3. 添加工具脚本..."
git add preprocess_data.py

echo "4. 添加文档..."
git add PPO_USAGE.md

echo "5. 添加示例脚本..."
git add example_usage.sh

# 6. 添加修改的文件
echo "6. 添加修改的文件..."
git add README.md
git add networks.py
git add environment.py
git add data_loader.py
git add agent.py

echo ""
echo "======================================"
echo "Git 状态预览:"
echo "======================================"
git status

echo ""
echo "======================================"
echo "提交信息预览:"
echo "======================================"
echo ""
cat << 'COMMITMSG'
feat: 引入 PPO 算法并修复 SAC 数值稳定性问题

主要变更:
1. 新增 PPO (Proximal Policy Optimization) 算法实现
   - ppo_agent.py: PPO 核心实现，支持 GAE 和 Dirichlet 分布
   - trainer_ppo.py: PPO 滚动训练器
   - main_ppo.py: PPO 训练入口
   - config_ppo.py: PPO 超参数配置

2. 数值稳定性修复
   - 使用 Dirichlet 分布替代 Softmax+log，避免 log(0)=-inf
   - 添加奖励缩放 (100x)，解决梯度消失
   - 添加 Action Masking，停牌/跌停股票权重强制为 0
   - 添加数据清洗，检查并修复 NaN/Inf

3. 新增工具
   - preprocess_data.py: 独立数据预处理脚本
   - PPO_USAGE.md: PPO 使用指南
   - example_usage.sh: 常用命令示例

4. 更新文档
   - README.md: 全面更新，强调 PPO 为推荐算法
   - 添加算法对比表 (PPO vs SAC)
   - 添加故障排除指南

算法特性:
- On-policy 策略，使用 Rollout Buffer
- Dirichlet 分布生成投资组合权重
- GAE (Generalized Advantage Estimation)
- Advantage 归一化
- 熵系数线性衰减 (0.01→0)
- 增量学习支持

PPO 相比 SAC 的优势:
- 更稳定的训练过程
- 解决了梯度消失问题
- 更好的数值稳定性
- 更适合金融序列决策

相关文档:
- PPO_USAGE.md
- NA_FIX_SUMMARY.md
COMMITMSG

echo ""
echo "======================================"
echo "执行命令:"
echo "git commit -m 'feat: 引入 PPO 算法...'"
echo "======================================"
