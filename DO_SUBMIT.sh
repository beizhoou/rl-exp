#!/bin/bash
# 一键提交脚本

echo "=========================================="
echo "Git 一键提交 - PPO 算法升级"
echo "=========================================="
echo ""

# 1. 添加 .gitignore
echo "[1/4] 添加 .gitignore..."
git add .gitignore

# 2. 添加所有源代码文件
echo "[2/4] 添加源代码文件..."
git add -f \
  config_ppo.py \
  ppo_agent.py \
  trainer_ppo.py \
  main_ppo.py \
  preprocess_data.py \
  PPO_USAGE.md \
  NA_FIX_SUMMARY.md \
  example_usage.sh \
  README.md \
  networks.py \
  environment.py \
  data_loader.py \
  agent.py \
  trainer.py \
  main.py \
  config.py \
  utils.py \
  benchmarks.py \
  run_benchmarks.py \
  broker_weights.py

# 3. 查看状态
echo ""
echo "[3/4] Git 状态:"
git status --short

# 4. 提交
echo ""
echo "[4/4] 提交..."
git commit -F COMMIT_MESSAGE.txt

echo ""
echo "=========================================="
echo "✅ 提交完成!"
echo "=========================================="
echo ""
echo "如需推送到远程:"
echo "  git push origin master"
