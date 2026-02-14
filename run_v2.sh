#!/bin/bash
# PPO V2 快速启动脚本
# 修复负奖励问题的改进版本

set -e

echo "=========================================="
echo "PPO V2 - A-Share Portfolio Management"
echo "=========================================="

# 检查参数
MODE=${1:-debug}

# 根据模式选择配置
case $MODE in
    debug)
        echo "🐛 运行调试模式（0成本 + 纯收益奖励）"
        python main_ppo_v2.py \
            --debug-zero-cost \
            --enable-cash \
            --reward-mode profit_only \
            --max-windows 2 \
            --batch-size 256 \
            --policy-type softmax_gaussian
        ;;
    
    curriculum)
        echo "🎓 运行课程学习模式"
        python main_ppo_v2.py \
            --enable-cash \
            --curriculum \
            --policy-type softmax_gaussian \
            --top-k 20 \
            --batch-size 512
        ;;
    
    quick)
        echo "⚡ 运行快速测试模式（2个窗口）"
        python main_ppo_v2.py \
            --debug-zero-cost \
            --enable-cash \
            --reward-mode risk_adjusted \
            --turnover-penalty 0.3 \
            --max-windows 2 \
            --batch-size 512 \
            --policy-type softmax_gaussian \
            --top-k 10
        ;;
    
    full)
        echo "🚀 运行完整训练模式"
        python main_ppo_v2.py \
            --enable-cash \
            --reward-mode risk_adjusted \
            --turnover-penalty 0.3 \
            --risk-penalty 10.0 \
            --transaction-cost 0.0015 \
            --policy-type softmax_gaussian \
            --top-k 10 \
            --batch-size 512
        ;;
    
    zero-cost-test)
        echo "🧪 0成本验证模式（验证模型能否学会正收益）"
        python main_ppo_v2.py \
            --debug-zero-cost \
            --enable-cash \
            --reward-mode risk_adjusted \
            --turnover-penalty 0.1 \
            --max-windows 3 \
            --batch-size 512 \
            --policy-type softmax_gaussian
        ;;
    
    dirichlet-test)
        echo "📊 Dirichlet策略对比测试"
        python main_ppo_v2.py \
            --debug-zero-cost \
            --enable-cash \
            --reward-mode profit_only \
            --max-windows 2 \
            --batch-size 256 \
            --policy-type dirichlet
        ;;
    
    no-cash)
        echo "🚫 无现金仓位模式（对比测试）"
        python main_ppo_v2.py \
            --debug-zero-cost \
            --no-cash \
            --reward-mode profit_only \
            --max-windows 2 \
            --batch-size 256
        ;;
    
    help|--help|-h)
        echo "使用方法: ./run_v2.sh [MODE]"
        echo ""
        echo "可用模式:"
        echo "  debug          - 调试模式（推荐首次运行）"
        echo "  curriculum     - 课程学习模式"
        echo "  quick          - 快速测试（2个窗口）"
        echo "  full           - 完整训练"
        echo "  zero-cost-test - 0成本验证"
        echo "  dirichlet-test - Dirichlet策略对比"
        echo "  no-cash        - 无现金仓位对比"
        echo "  help           - 显示此帮助"
        echo ""
        echo "示例:"
        echo "  ./run_v2.sh debug      # 首次运行推荐"
        echo "  ./run_v2.sh full       # 正式训练"
        echo "  ./run_v2.sh quick      # 快速验证"
        ;;
    
    *)
        echo "❌ 未知模式: $MODE"
        echo "请使用: ./run_v2.sh help 查看可用模式"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
