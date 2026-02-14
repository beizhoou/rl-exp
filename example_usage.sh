#!/bin/bash

# ============================================================
# 滚动微调步进回测训练方案 - 使用示例
# Rolling Walk-Forward with Fine-tuning - Usage Examples
# ============================================================

cd /root/autodl-tmp/exper_rl

# -----------------------------------------------------------
# 0. 数据预处理（推荐首次执行）
#    只需执行一次，之后可重复使用
# -----------------------------------------------------------
echo "=== 方案0: 数据预处理（提高后续训练速度）==="

# 基础预处理
python preprocess_data.py

# 指定输出路径
python preprocess_data.py -o ./data/processed_data.pkl

# 指定日期范围
python preprocess_data.py \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    -o ./data/processed_2020_2023.pkl

# 验证预处理文件
python preprocess_data.py --verify processed_data.pkl

# -----------------------------------------------------------
# 1. 完整训练（推荐 ⚡️ 使用预处理数据）
#    24月训练 + 1月验证 + 1月测试，滚动步进
# -----------------------------------------------------------
echo "=== 方案1: 完整滚动训练（使用预处理数据）==="
python main.py --preprocessed-data processed_data.pkl --device cuda --run-benchmarks

# -----------------------------------------------------------
# 2. 快速实验（缩短训练轮数）
#    适合调试和快速验证
# -----------------------------------------------------------
echo "=== 方案2: 快速实验（使用预处理数据）==="
python main.py \
    --preprocessed-data processed_data.pkl \
    --device cuda \
    --episodes-first 100 \
    --episodes-finetune 20 \
    --run-benchmarks

# -----------------------------------------------------------
# 3. 禁用增量学习（对比实验）
#    每轮都重新初始化权重，不继承上一轮
# -----------------------------------------------------------
echo "=== 方案3: 禁用增量学习（消融实验）==="
python main.py \
    --preprocessed-data processed_data.pkl \
    --device cuda \
    --no-inherit \
    --run-benchmarks

# -----------------------------------------------------------
# 4. 仅运行基准策略对比
#    不训练RL，只跑基准策略
# -----------------------------------------------------------
echo "=== 方案4: 仅基准对比（使用预处理数据）==="
python main.py --preprocessed-data processed_data.pkl --benchmark-only

# -----------------------------------------------------------
# 5. 原始数据流（首次使用或数据更新）
#    包含完整的数据预处理流程
# -----------------------------------------------------------
echo "=== 方案5: 原始数据流（首次使用）==="
python main.py --device cuda --run-benchmarks

# 同时保存预处理数据供下次使用
python main.py \
    --device cuda \
    --save-preprocessed processed_data.pkl \
    --run-benchmarks

# -----------------------------------------------------------
# 6. 仅执行数据预处理（不训练）
# -----------------------------------------------------------
echo "=== 方案6: 仅数据预处理 ==="
python main.py --preprocess-only --save-preprocessed processed_data.pkl

# -----------------------------------------------------------
# 7. 调整batch size（大显存显卡）
# -----------------------------------------------------------
echo "=== 方案7: 大batch训练（使用预处理数据）==="
python main.py \
    --preprocessed-data processed_data.pkl \
    --device cuda \
    --batch-size 512 \
    --episodes-first 500 \
    --run-benchmarks

# -----------------------------------------------------------
# 8. 指定日期范围训练
# -----------------------------------------------------------
echo "=== 方案8: 指定日期范围（使用预处理数据）==="
python main.py \
    --preprocessed-data processed_data.pkl \
    --device cuda \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --run-benchmarks

# -----------------------------------------------------------
# TensorBoard监控
# -----------------------------------------------------------
echo "=== 启动TensorBoard ==="
echo "tensorboard --logdir=./logs --port=6006"

# -----------------------------------------------------------
# 查看结果
# -----------------------------------------------------------
echo "=== 查看结果 ==="
echo "基准对比汇总: cat benchmark_results/benchmark_summary.csv"
echo "训练汇总图:   plots/summary.png"
echo "窗口时间线:   plots/window_timeline.png"
echo "TensorBoard:  tensorboard --logdir=./logs"
