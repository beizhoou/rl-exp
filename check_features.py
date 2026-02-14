#!/usr/bin/env python3
"""
检查哪些特征会有 std=0（所有股票值相同）
"""
import pickle
import numpy as np
import pandas as pd

# 加载预处理数据
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['df']
feature_cols = data['feature_cols']

print("="*70)
print("特征分析 - 查找 std=0 的特征")
print("="*70)

# 选取某一天的数据进行分析
dates = df['date'].unique()
sample_date = dates[20]  # 第20天（对应之前的警告）
day_data = df[df['date'] == sample_date]

print(f"\n分析日期: {sample_date}")
print(f"股票数量: {len(day_data)}")

# 计算每个特征的统计量
feature_stats = []
for col in feature_cols:
    if col in day_data.columns:
        values = day_data[col]
        mean = values.mean()
        std = values.std()
        min_val = values.min()
        max_val = values.max()
        unique_count = values.nunique()
        
        feature_stats.append({
            'feature': col,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'unique': unique_count,
            'is_constant': std == 0 or unique_count == 1
        })

stats_df = pd.DataFrame(feature_stats)

# 找出恒定特征
constant_features = stats_df[stats_df['is_constant'] == True]
print(f"\n🔍 恒定特征 (std=0 或只有一个唯一值): {len(constant_features)} 个")
if len(constant_features) > 0:
    print(constant_features[['feature', 'mean', 'std', 'unique']].to_string(index=False))

# 找出接近恒定的特征 (std < 0.01)
near_constant = stats_df[(stats_df['std'] < 0.01) & (stats_df['std'] > 0)]
print(f"\n🔍 接近恒定特征 (std < 0.01): {len(near_constant)} 个")
if len(near_constant) > 0:
    print(near_constant[['feature', 'mean', 'std', 'unique']].head(10).to_string(index=False))

# 检查 pad 特征
pad_features = [f for f in feature_cols if f.startswith('pad_')]
print(f"\n🔍 填充特征 (pad_*): {len(pad_features)} 个")
print(f"   这些特征全部填充为0，因此std=0")

# 检查 guba 特征
guba_features = [f for f in feature_cols if f.startswith('guba_')]
print(f"\n🔍 股吧特征 (guba_*): {len(guba_features)} 个")
for f in guba_features:
    row = stats_df[stats_df['feature'] == f].iloc[0]
    print(f"   {f}: std={row['std']:.4f}, unique={row['unique']}")

# 检查 vlm 特征
vlm_features = [f for f in feature_cols if f.startswith('vlm_')]
print(f"\n🔍 VLM特征 (vlm_*): {len(vlm_features)} 个")
for f in vlm_features:
    row = stats_df[stats_df['feature'] == f].iloc[0]
    print(f"   {f}: std={row['std']:.4f}, unique={row['unique']}")

# 检查 fundamental 特征
fund_features = ['pe_ttm', 'pb', 'peg', 'ps', 'roe', 'growth', 'gross_margin', 'debt_ratio']
print(f"\n🔍 基本面特征: {len(fund_features)} 个")
for f in fund_features:
    if f in stats_df['feature'].values:
        row = stats_df[stats_df['feature'] == f].iloc[0]
        print(f"   {f}: std={row['std']:.4f}, unique={row['unique']}")

print("\n" + "="*70)
print("分析完成")
print("="*70)
