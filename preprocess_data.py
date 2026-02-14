#!/usr/bin/env python3
"""
数据预处理脚本
==============
将多源异构数据预处理并保存，避免训练时重复处理

使用方法:
    # 基础预处理
    python preprocess_data.py
    
    # 指定输出路径
    python preprocess_data.py --output /path/to/processed_data.pkl
    
    # 指定日期范围
    python preprocess_data.py --start-date 2020-01-01 --end-date 2023-12-31
    
    # 验证预处理文件
    python preprocess_data.py --verify /path/to/processed_data.pkl

输出文件:
    - processed_data.pkl: 完整预处理数据（包含特征工程和标准化统计量）
    - processed_data.csv: CSV格式（用于外部分析）
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional

# 导入项目配置
from config import data_cfg, print_config
from data_loader import MultiSourceDataLoader


def save_processed_data(df: pd.DataFrame, 
                        feature_cols: list,
                        feature_stats: dict,
                        output_path: str,
                        metadata: dict = None):
    """
    保存预处理后的数据
    
    Args:
        df: 预处理后的DataFrame
        feature_cols: 特征列名列表
        feature_stats: 特征统计量（用于后续标准化）
        output_path: 输出路径
        metadata: 元数据信息
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else '.', 
                exist_ok=True)
    
    # 准备数据包
    data_package = {
        'df': df,
        'feature_cols': feature_cols,
        'feature_stats': feature_stats,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat(),
        'config': {
            'n_stocks': data_cfg.n_stocks,
            'n_features': data_cfg.n_features,
            'lookback_window': data_cfg.lookback_window,
            'start_date': data_cfg.start_date,
            'end_date': data_cfg.end_date,
        }
    }
    
    # 保存为pickle（保留完整数据结构）
    with open(output_path, 'wb') as f:
        pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n💾 预处理数据已保存到: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 同时保存CSV版本（用于外部分析）
    csv_path = output_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"   CSV版本: {csv_path}")
    
    return output_path


def load_processed_data(input_path: str) -> Dict:
    """
    加载预处理后的数据
    
    Args:
        input_path: 预处理文件路径
        
    Returns:
        包含df、feature_cols、feature_stats等的字典
    """
    print(f"\n📂 加载预处理数据: {input_path}")
    
    with open(input_path, 'rb') as f:
        data_package = pickle.load(f)
    
    print(f"   ✅ 创建于: {data_package.get('created_at', 'Unknown')}")
    print(f"   ✅ 数据行数: {len(data_package['df']):,}")
    print(f"   ✅ 特征维度: {len(data_package['feature_cols'])}")
    
    return data_package


def verify_processed_data(input_path: str):
    """验证预处理数据文件的完整性"""
    print(f"\n{'='*70}")
    print("🔍 验证预处理数据文件")
    print(f"{'='*70}")
    
    try:
        data = load_processed_data(input_path)
        
        # 检查必要字段
        required_keys = ['df', 'feature_cols', 'feature_stats', 'metadata']
        for key in required_keys:
            if key not in data:
                print(f"   ❌ 缺失字段: {key}")
                return False
        
        # 验证数据
        df = data['df']
        feature_cols = data['feature_cols']
        
        print(f"\n📊 数据验证:")
        print(f"   股票数量: {df[data_cfg.stock_col].nunique()}")
        print(f"   日期范围: {df[data_cfg.date_col].min()} ~ {df[data_cfg.date_col].max()}")
        print(f"   总行数: {len(df):,}")
        print(f"   特征数: {len(feature_cols)}")
        
        # 检查特征列
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"   ⚠️  缺失特征列: {missing_cols}")
        else:
            print(f"   ✅ 所有特征列存在")
        
        # 检查缺失值
        missing_ratio = df[feature_cols].isna().mean().mean()
        print(f"   特征缺失率: {missing_ratio:.4%}")
        
        # 统计量检查
        stats = data['feature_stats']
        print(f"\n📈 特征统计量:")
        print(f"   均值范围: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
        print(f"   标准差范围: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")
        
        print(f"\n✅ 验证通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_data(output_path: str = 'processed_data.pkl',
                    start_date: str = None,
                    end_date: str = None,
                    save_stats: bool = True) -> str:
    """
    执行完整的数据预处理流程
    
    Args:
        output_path: 输出文件路径
        start_date: 开始日期（覆盖配置）
        end_date: 结束日期（覆盖配置）
        save_stats: 是否保存统计量
        
    Returns:
        输出文件路径
    """
    print(f"\n{'='*70}")
    print("🚀 数据预处理")
    print(f"{'='*70}")
    
    # 打印配置
    print_config()
    
    # 应用日期范围覆盖
    if start_date:
        data_cfg.start_date = start_date
    if end_date:
        data_cfg.end_date = end_date
    
    print(f"\n📅 数据日期范围: {data_cfg.start_date} ~ {data_cfg.end_date}")
    
    # 步骤1: 加载多源数据
    print(f"\n{'='*70}")
    print("📥 步骤 1/4: 加载多源数据")
    print(f"{'='*70}")
    
    loader = MultiSourceDataLoader(data_cfg)
    df = loader.load_and_merge()
    
    if len(df) == 0:
        raise ValueError("数据加载失败，请检查数据路径和格式")
    
    # 步骤2: 特征工程
    print(f"\n{'='*70}")
    print("🔧 步骤 2/4: 特征工程")
    print(f"{'='*70}")
    
    # 识别特征列
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    print(f"   识别到 {len(feature_cols)} 个特征列")
    
    # 步骤3: 计算全局统计量（用于后续标准化）
    print(f"\n{'='*70}")
    print("📊 步骤 3/4: 计算特征统计量")
    print(f"{'='*70}")
    
    if save_stats:
        # 计算全局统计量（用于Z-score标准化）
        feature_stats = {
            'mean': df[feature_cols].mean(),
            'std': df[feature_cols].std() + 1e-8,
            'min': df[feature_cols].min(),
            'max': df[feature_cols].max(),
            'median': df[feature_cols].median(),
        }
        
        print(f"   已计算 {len(feature_cols)} 个特征的统计量")
    else:
        feature_stats = {}
    
    # 步骤4: 保存数据
    print(f"\n{'='*70}")
    print("💾 步骤 4/4: 保存预处理数据")
    print(f"{'='*70}")
    
    metadata = {
        'n_rows': len(df),
        'n_stocks': df[data_cfg.stock_col].nunique(),
        'n_features': len(feature_cols),
        'date_range': {
            'min': df[data_cfg.date_col].min().isoformat(),
            'max': df[data_cfg.date_col].max().isoformat(),
        },
        'columns': list(df.columns),
        'feature_cols': feature_cols,
    }
    
    save_processed_data(df, feature_cols, feature_stats, output_path, metadata)
    
    print(f"\n{'='*70}")
    print("✅ 数据预处理完成!")
    print(f"{'='*70}")
    print(f"\n预处理文件: {output_path}")
    print(f"\n使用方法:")
    print(f"   from preprocess_data import load_processed_data")
    print(f"   data = load_processed_data('{output_path}')")
    print(f"   df = data['df']")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal RL Trading - Data Preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础预处理
    python preprocess_data.py
    
    # 指定输出路径
    python preprocess_data.py -o ./data/processed.pkl
    
    # 指定日期范围
    python preprocess_data.py --start-date 2020-01-01 --end-date 2023-12-31
    
    # 验证预处理文件
    python preprocess_data.py --verify ./processed_data.pkl
        """
    )
    
    parser.add_argument('-o', '--output', type=str, default='processed_data.pkl',
                       help='输出文件路径 (默认: processed_data.pkl)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--verify', type=str, default=None,
                       help='验证指定的预处理文件')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据根目录（覆盖配置中的路径）')
    
    args = parser.parse_args()
    
    # 验证模式
    if args.verify:
        verify_processed_data(args.verify)
        return
    
    # 覆盖数据路径（如果指定）
    if args.data_dir:
        data_cfg.guba_dir = os.path.join(args.data_dir, 'guba_sentiment_results')
        data_cfg.basic_dir = os.path.join(args.data_dir, 'stcok_basic')
        data_cfg.vlm_file = os.path.join(args.data_dir, 'vlm_sentiment_analysis.csv')
    
    # 执行预处理
    try:
        output_path = preprocess_data(
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # 自动验证
        print(f"\n自动验证...")
        verify_processed_data(output_path)
        
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
