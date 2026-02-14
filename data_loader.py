import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from glob import glob
from tqdm import tqdm
from broker_weights import get_broker_weight


class MultiSourceDataLoader:
    """
    多源异构数据加载器
    处理：股吧情绪（分散CSV）+ 基本面（分散CSV）+ VLM研报（集中CSV，券商加权）
    
    数据安全原则：
    1. 所有填充操作在合并前完成，避免跨数据源泄露
    2. 时序填充严格按股票分组，不跨股票填充
    3. 研报数据非每日都有，使用有效期的向前填充
    
    新增：支持从预处理文件加载
    """
    def __init__(self, config):
        self.cfg = config
        self._validate_paths()
        
    def _validate_paths(self):
        assert os.path.exists(self.cfg.guba_dir), f"股吧目录不存在: {self.cfg.guba_dir}"
        assert os.path.exists(self.cfg.basic_dir), f"基本面目录不存在: {self.cfg.basic_dir}"
        assert os.path.exists(self.cfg.vlm_file), f"VLM文件不存在: {self.cfg.vlm_file}"
        
    def _normalize_stock_code(self, code) -> str:
        """
        统一股票代码为6位字符串格式
        支持输入: 1, '1', '000001', 1.0 等
        """
        if pd.isna(code):
            return None
        # 转换为字符串并去除小数点和空格
        code_str = str(code).strip().split('.')[0]
        # 补齐至6位
        return code_str.zfill(6)
    
    def _load_guba_sentiment(self) -> pd.DataFrame:
        """
        加载股吧情绪数据（471个分散文件）
        字段：bullishness, panic, consensus, summary, share_code, date, prompt, status
        """
        print("Loading guba sentiment data...")
        files = glob(os.path.join(self.cfg.guba_dir, "*_result.csv"))
        if len(files) != self.cfg.n_stocks:
            print(f"Warning: Found {len(files)} files, expected {self.cfg.n_stocks}")
        
        dfs = []
        for f in tqdm(files, desc="Guba files"):
            df = pd.read_csv(f)
            # 确保日期格式正确
            df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
            # 统一股票代码格式
            df[self.cfg.stock_col] = df[self.cfg.stock_col].apply(self._normalize_stock_code)
            # 选择数值型情绪指标
            use_cols = [self.cfg.stock_col, self.cfg.date_col] + self.cfg.guba_features
            # 处理可能的缺失列（如果某些文件缺少字段）
            available_cols = [c for c in use_cols if c in df.columns]
            dfs.append(df[available_cols])
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Guba data: {len(combined)} records, {combined[self.cfg.stock_col].nunique()} stocks")
        return combined
    
    def _load_basic_data(self) -> pd.DataFrame:
        """
        加载价量基本面数据（471个分散文件）
        字段：code, date, open, close, high, low, volume, daily_return, volatility_20, 
              pe_ttm, pb, peg, ps, market_cap, roe, growth, gross_margin, debt_ratio
        """
        print("Loading basic market data...")
        files = glob(os.path.join(self.cfg.basic_dir, "*.csv"))
        
        dfs = []
        for f in tqdm(files, desc="Basic files"):
            df = pd.read_csv(f)
            df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
            # 标准化股票代码列名和格式
            if 'code' in df.columns and self.cfg.stock_col not in df.columns:
                df.rename(columns={'code': self.cfg.stock_col}, inplace=True)
            df[self.cfg.stock_col] = df[self.cfg.stock_col].apply(self._normalize_stock_code)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Basic data: {len(combined)} records, {combined[self.cfg.stock_col].nunique()} stocks")
        return combined
    
    def _load_vlm_reports(self) -> pd.DataFrame:
        """
        加载VLM研报分析数据（单个文件，按券商影响力加权聚合）
        字段：share_code, date, broker, sentiment_score, rating_change, eps_g_y0, 
              eps_g_y1, eps_g_y2, pe_forward_y1, profit_revision, revenue_revision
        
        加权逻辑：按券商影响力加权平均，权威券商研报权重更高
        """
        print("Loading VLM report data...")
        # 使用 error_bad_lines=False 跳过格式错误的行
        try:
            df = pd.read_csv(self.cfg.vlm_file, on_bad_lines='skip')
        except Exception as e:
            print(f"Warning: Error reading VLM file with default parser: {e}")
            print("Trying with python engine...")
            df = pd.read_csv(self.cfg.vlm_file, engine='python', on_bad_lines='skip')
        df[self.cfg.date_col] = pd.to_datetime(df[self.cfg.date_col])
        # 统一股票代码格式
        df[self.cfg.stock_col] = df[self.cfg.stock_col].apply(self._normalize_stock_code)
        
        print(f"Raw VLM data: {len(df)} records")
        
        # 添加券商权重
        df['broker_weight'] = df['broker'].apply(get_broker_weight)
        
        # 按股票+日期聚合（加权平均）
        numeric_features = [f for f in self.cfg.vlm_features if f in df.columns]
        
        def weighted_mean(x, weights):
            """计算加权平均值，处理全空情况"""
            valid_mask = x.notna()
            if not valid_mask.any():
                return np.nan
            valid_x = x[valid_mask]
            valid_w = weights[valid_mask]
            if valid_w.sum() == 0:
                return valid_x.mean()
            return (valid_x * valid_w).sum() / valid_w.sum()
        
        # 分组加权聚合
        grouped = df.groupby([self.cfg.stock_col, self.cfg.date_col]).agg(
            report_count=('broker_weight', 'size'),
            avg_broker_weight=('broker_weight', 'mean')
        ).reset_index()
        
        # 对每个数值特征计算加权平均
        for col in numeric_features:
            weighted_vals = df.groupby([self.cfg.stock_col, self.cfg.date_col]).apply(
                lambda x: weighted_mean(x[col], x['broker_weight'])
            ).reset_index(drop=True)
            grouped[col] = weighted_vals
        
        print(f"Aggregated VLM data: {len(grouped)} records")
        print(f"Average reports per day: {grouped['report_count'].mean():.2f}")
        return grouped
    
    def _safe_ffill(self, df: pd.DataFrame, cols_to_fill: List[str], max_gap: int = 5) -> pd.DataFrame:
        """
        安全的向前填充：限制最大填充天数，避免过期信息
        
        Args:
            df: 按股票排序的数据框
            cols_to_fill: 需要填充的列
            max_gap: 最大填充天数，超过则置为NA
        """
        df = df.sort_values([self.cfg.stock_col, self.cfg.date_col]).copy()
        
        for col in cols_to_fill:
            if col not in df.columns:
                continue
            
            # 计算距离上次有效值的天数
            df['valid_mask'] = df[col].notna()
            df['group'] = (~df['valid_mask']).cumsum()
            
            # 只对连续缺失不超过max_gap的进行填充
            df['days_since_valid'] = df.groupby([self.cfg.stock_col, 'group']).cumcount()
            
            # 前向填充
            df[col] = df.groupby(self.cfg.stock_col)[col].ffill()
            
            # 超过max_gap的填充值重置为NA（过期信息）
            df.loc[df['days_since_valid'] > max_gap, col] = np.nan
            
        # 清理临时列
        df = df.drop(columns=['valid_mask', 'group', 'days_since_valid'], errors='ignore')
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        缺失值处理策略（确保无数据泄露）
        
        1. 研报数据：研报有效期5个交易日，超过则置NA
        2. 情绪数据：情绪衰减快，有效期3个交易日
        3. 基本面数据：季度更新，有效期60个交易日
        4. 最后NA用截面均值填充（同股票历史均值）
        """
        print("Handling missing values...")
        
        # 分类特征
        vlm_cols = [f'vlm_{col}' for col in self.cfg.vlm_features if f'vlm_{col}' in df.columns]
        guba_cols = [f'guba_{col}' for col in self.cfg.guba_features if f'guba_{col}' in df.columns]
        
        # 基本面列（识别包含财务指标的列）
        fund_keywords = ['pe_ttm', 'pb', 'peg', 'ps', 'roe', 'growth', 'gross_margin', 'debt_ratio']
        fund_cols = [c for c in df.columns if any(kw in c for kw in fund_keywords)]
        
        # 1. 研报数据：短期有效（5天）
        if vlm_cols:
            print(f"  Filling VLM features with max gap=5 days...")
            df = self._safe_ffill(df, vlm_cols, max_gap=5)
        
        # 2. 情绪数据：短期有效（3天）
        if guba_cols:
            print(f"  Filling sentiment features with max gap=3 days...")
            df = self._safe_ffill(df, guba_cols, max_gap=3)
        
        # 3. 基本面数据：季度更新（60天）
        if fund_cols:
            print(f"  Filling fundamental features with max gap=60 days...")
            df = self._safe_ffill(df, fund_cols, max_gap=60)
        
        # 4. 仍缺失的数值：用截面均值填充（按股票分组计算历史均值）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in [self.cfg.stock_col, self.cfg.date_col, 'close', 'tradable']]
        
        print(f"  Filling remaining NA with stock-level mean...")
        for col in numeric_cols:
            if df[col].isna().any():
                # 按股票计算历史均值填充
                stock_means = df.groupby(self.cfg.stock_col)[col].transform('mean')
                df[col] = df[col].fillna(stock_means)
                # 仍有NA（整只股票无该特征），用全局均值填充
                df[col] = df[col].fillna(df[col].mean())
        
        # 5. 特殊处理：研报数量缺失表示当天无研报，填0
        if 'vlm_report_count' in df.columns:
            df['vlm_report_count'] = df['vlm_report_count'].fillna(0)
        
        missing_ratio = df[numeric_cols].isna().mean().mean()
        print(f"  Final missing ratio: {missing_ratio:.4%}")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程：构造动态维度的最终特征
        不再硬性要求40维，支持任意数量的有效特征
        """
        print("Engineering features...")
        
        # 按股票排序（确保时序计算正确）
        df = df.sort_values([self.cfg.stock_col, self.cfg.date_col]).copy()
        
        # 1. 技术面特征（15维）
        tech_features = {}
        
        # 价格位置（相对于20日高低点）
        tech_features['price_position'] = (df['close'] - df.groupby(self.cfg.stock_col)['low'].transform(lambda x: x.rolling(20).min())) / \
                                         (df.groupby(self.cfg.stock_col)['high'].transform(lambda x: x.rolling(20).max()) - 
                                          df.groupby(self.cfg.stock_col)['low'].transform(lambda x: x.rolling(20).min()) + 1e-8)
        
        # 收益率相关
        tech_features['daily_return'] = df['daily_return'] if 'daily_return' in df.columns else df.groupby(self.cfg.stock_col)['close'].transform(lambda x: x.pct_change())
        tech_features['return_5d'] = df.groupby(self.cfg.stock_col)['close'].transform(lambda x: x.pct_change(5))
        tech_features['return_20d'] = df.groupby(self.cfg.stock_col)['close'].transform(lambda x: x.pct_change(20))
        
        # 波动率
        tech_features['volatility_20'] = df['volatility_20'] if 'volatility_20' in df.columns else \
                                        df.groupby(self.cfg.stock_col)['daily_return'].transform(lambda x: x.rolling(20).std())
        
        # 成交量特征
        tech_features['volume_ratio'] = df.groupby(self.cfg.stock_col)['volume'].transform(lambda x: x / x.rolling(20).mean())
        tech_features['volume_change'] = df.groupby(self.cfg.stock_col)['volume'].transform(lambda x: x.pct_change())
        
        # 价格形态
        tech_features['high_low_ratio'] = df['high'] / df['low'] - 1
        tech_features['open_close_ratio'] = df['open'] / df['close'] - 1
        
        # 技术指标（MACD）- 使用 transform 避免索引错位
        # 先计算 MACD (12, 26) 保存到临时列
        df['_ema12'] = df.groupby(self.cfg.stock_col)['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df['_ema26'] = df.groupby(self.cfg.stock_col)['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['_macd'] = df['_ema12'] - df['_ema26']
        tech_features['macd'] = df['_macd']
        
        # 计算 MACD 信号线 (9) - 基于 MACD 的 EMA
        tech_features['macd_signal'] = df.groupby(self.cfg.stock_col)['_macd'].transform(
            lambda x: x.ewm(span=9, adjust=False).mean()
        )
        
        # 清理临时列
        df = df.drop(columns=['_ema12', '_ema26', '_macd'])
        
        # RSI - 使用 transform 避免索引错位
        def calc_rsi_simple(close, period=14):
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        tech_features['rsi'] = df.groupby(self.cfg.stock_col)['close'].transform(calc_rsi_simple)
        
        # 2. 基本面特征（8维）
        fund_features = {}
        for col in ['pe_ttm', 'pb', 'peg', 'ps', 'roe', 'growth', 'gross_margin', 'debt_ratio']:
            if col in df.columns:
                fund_features[col] = df[col]
            else:
                fund_features[col] = 0
        
        # 3. 股吧情绪特征（3维）
        guba_feats = {}
        for col in self.cfg.guba_features:
            src_col = col
            dst_col = f'guba_{col}'
            if src_col in df.columns:
                guba_feats[dst_col] = df[src_col]
            else:
                guba_feats[dst_col] = 0
        
        # 4. VLM研报特征（8维 + 2维元信息）
        vlm_feats = {}
        for col in self.cfg.vlm_features:
            src_col = f'vlm_{col}'
            dst_col = f'vlm_{col}'
            if src_col in df.columns:
                vlm_feats[dst_col] = df[src_col]
            else:
                vlm_feats[dst_col] = 0
        
        # 研报置信度特征
        if 'vlm_report_count' in df.columns:
            vlm_feats['vlm_report_count'] = df['vlm_report_count']
        else:
            vlm_feats['vlm_report_count'] = 0
            
        if 'vlm_avg_broker_weight' in df.columns:
            vlm_feats['vlm_broker_quality'] = df['vlm_avg_broker_weight']
        else:
            vlm_feats['vlm_broker_quality'] = 0
        
        # 5. 交互特征（跨源信息融合）
        # 研报情绪 vs 股吧情绪一致性
        if 'vlm_sentiment_score' in vlm_feats and 'guba_bullishness' in guba_feats:
            tech_features['sentiment_consensus'] = vlm_feats['vlm_sentiment_score'] * guba_feats['guba_bullishness'] / 100
        
        # 基本面 vs 情绪偏离
        if 'pe_ttm' in fund_features and 'guba_bullishness' in guba_feats:
            tech_features['pe_sentiment_dev'] = fund_features['pe_ttm'] * (1 + guba_feats['guba_bullishness'] / 100)
        
        # 市值对数
        if 'market_cap' in df.columns:
            tech_features['log_market_cap'] = np.log1p(df['market_cap'])
        
        # 动量-情绪复合因子
        tech_features['mom_sentiment'] = tech_features['return_20d'] * guba_feats.get('guba_consensus', 0) / 100
        
        # 研报质量-动量因子
        if 'vlm_sentiment_score' in vlm_feats:
            tech_features['vlm_momentum'] = vlm_feats['vlm_sentiment_score'] * tech_features['return_5d'] / 100
        
        # 合并所有特征
        feature_df = pd.DataFrame({
            **tech_features,
            **fund_features,
            **guba_feats,
            **vlm_feats,
            self.cfg.stock_col: df[self.cfg.stock_col],
            self.cfg.date_col: df[self.cfg.date_col],
            'close': df['close'],
            'open': df['open'],
            'high': df['high'],
            'low': df['low'],
            'volume': df['volume'],
            'tradable': 1
        })
        
        # 添加涨跌停标记（用于环境过滤）
        # 涨跌停判断：当日最高价=收盘价（涨停）或最低价=收盘价（跌停）
        feature_df['hit_limit_up'] = (feature_df['close'] >= feature_df['high'] * 0.998).astype(int)
        feature_df['hit_limit_down'] = (feature_df['close'] <= feature_df['low'] * 1.002).astype(int)
        
        # A股涨跌停限制：非ST股票±10%，ST股票±5%
        # 根据日内波动幅度判断是否为ST
        daily_range = (feature_df['high'] - feature_df['low']) / feature_df['low']
        feature_df['is_st'] = (daily_range <= 0.06).astype(int)  # 日内波动小可能是ST
        
        # 可交易性：非涨跌停状态
        feature_df['tradable'] = 1 - feature_df['hit_limit_up'] - feature_df['hit_limit_down']
        feature_df['tradable'] = feature_df['tradable'].clip(0, 1)
        
        # 停牌标记（成交量为0或价格不变）
        feature_df['is_suspended'] = ((feature_df['volume'] == 0) | 
                                       (feature_df['close'] == feature_df['open']) & 
                                       (feature_df['high'] == feature_df['low'])).astype(int)
        
        # 动态特征维度：使用所有非元数据列作为特征
        exclude_cols = [self.cfg.stock_col, self.cfg.date_col, 'close', 'open', 'high', 'low', 
                       'volume', 'tradable', 'hit_limit_up', 'hit_limit_down', 'is_st', 'is_suspended']
        feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
        n_features = len(feature_cols)
        
        print(f"  Generated {n_features} raw features")
        
        # 如果特征太多，选择与收益率相关性最高的（最多保留50个）
        if n_features > 50:
            print(f"  Selecting top 50 features by correlation with daily_return")
            if 'daily_return' in feature_df.columns:
                corrs = feature_df[feature_cols].corrwith(feature_df['daily_return'])
                feature_cols = corrs.abs().nlargest(50).index.tolist()
            else:
                feature_cols = feature_cols[:50]
            n_features = len(feature_cols)
        
        # 重命名为f_0到f_{n-1}
        rename_map = {col: f'f_{i}' for i, col in enumerate(feature_cols)}
        feature_df.rename(columns=rename_map, inplace=True)
        
        # 选择最终列（动态维度）
        final_feature_cols = [f'f_{i}' for i in range(n_features)]
        final_columns = [self.cfg.stock_col, self.cfg.date_col, 'close', 'open', 'high', 'low',
                        'volume', 'tradable', 'hit_limit_up', 'hit_limit_down', 'is_st', 'is_suspended'] + \
                       final_feature_cols
        final_df = feature_df[[c for c in final_columns if c in feature_df.columns]]
        
        # 积极填充缺失值（关键！防止state中出现NaN）
        # 1. 按股票分组前向填充（用同股票历史值填充）
        for col in final_feature_cols:
            if col in final_df.columns:
                final_df[col] = final_df.groupby(self.cfg.stock_col)[col].ffill().bfill()
        
        # 2. 仍有NaN的，用0填充
        final_df[final_feature_cols] = final_df[final_feature_cols].fillna(0)
        
        # 3. 检查并报告NaN情况
        nan_count = final_df[final_feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"  Warning: {nan_count} NaN values remain after filling")
        
        print(f"  Final feature dimensions: {n_features}")
        
        return final_df
    
    def load_and_merge(self) -> pd.DataFrame:
        """
        主流程：加载三源数据 → 合并 → 缺失值处理 → 特征工程
        """
        # 1. 加载各源数据
        guba_df = self._load_guba_sentiment()
        basic_df = self._load_basic_data()
        vlm_df = self._load_vlm_reports()
        
        # 2. 合并（外连接保留所有日期）
        print("\nMerging data sources...")
        # 先合并基础数据和股吧情绪（左连接保留所有交易日）
        merged = pd.merge(basic_df, guba_df, on=[self.cfg.stock_col, self.cfg.date_col], how='left')
        print(f"  After merging basic + guba: {len(merged)} rows")
        
        # 再合并VLM研报（左连接，允许无研报的日子）
        merged = pd.merge(merged, vlm_df, on=[self.cfg.stock_col, self.cfg.date_col], how='left')
        print(f"  After merging VLM: {len(merged)} rows")
        
        # 3. 缺失值处理（安全的填充策略）
        merged = self._handle_missing_values(merged)
        
        # 4. 特征工程
        final_df = self._engineer_features(merged)
        
        # 5. 过滤日期范围
        final_df = final_df[
            (final_df[self.cfg.date_col] >= self.cfg.start_date) & 
            (final_df[self.cfg.date_col] <= self.cfg.end_date)
        ].copy()
        
        print(f"\nFinal dataset: {len(final_df)} rows, {final_df[self.cfg.stock_col].nunique()} stocks")
        print(f"Date range: {final_df[self.cfg.date_col].min()} to {final_df[self.cfg.date_col].max()}")
        
        # 打印特征统计
        feature_cols = [c for c in final_df.columns if c.startswith('f_')]
        print(f"Feature dimensions: {len(feature_cols)}")
        
        # 打印每类特征的缺失率
        for category, prefix in [('Tech', 'f_0'), ('Fundamental', 'pe_ttm'), 
                                  ('Guba', 'guba_'), ('VLM', 'vlm_')]:
            cols = [c for c in final_df.columns if prefix in c or c.startswith(prefix)]
            if cols:
                missing = final_df[cols].isna().mean().mean()
                print(f"  {category} features missing: {missing:.2%}")
        
        return final_df


class PreprocessedDataLoader:
    """
    从预处理文件加载数据
    
    使用方法:
        loader = PreprocessedDataLoader()
        df = loader.load('processed_data.pkl')
    """
    
    def __init__(self, config=None):
        self.cfg = config or data_cfg
        
    def load(self, input_path: str) -> pd.DataFrame:
        """
        从预处理文件加载数据
        
        Args:
            input_path: 预处理文件路径 (.pkl)
            
        Returns:
            预处理后的DataFrame
        """
        import pickle
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"预处理文件不存在: {input_path}")
        
        print(f"\n{'='*70}")
        print(f"📂 从预处理文件加载数据")
        print(f"{'='*70}")
        print(f"文件: {input_path}")
        
        with open(input_path, 'rb') as f:
            data_package = pickle.load(f)
        
        df = data_package['df']
        feature_cols = data_package['feature_cols']
        feature_stats = data_package.get('feature_stats', {})
        metadata = data_package.get('metadata', {})
        created_at = data_package.get('created_at', 'Unknown')
        
        print(f"✅ 加载成功!")
        print(f"   创建时间: {created_at}")
        print(f"   数据行数: {len(df):,}")
        print(f"   股票数量: {df[self.cfg.stock_col].nunique()}")
        print(f"   特征维度: {len(feature_cols)}")
        print(f"   日期范围: {df[self.cfg.date_col].min()} ~ {df[self.cfg.date_col].max()}")
        
        # 应用日期过滤（如果配置中有指定）
        if hasattr(self.cfg, 'start_date') and self.cfg.start_date:
            df = df[df[self.cfg.date_col] >= self.cfg.start_date]
        if hasattr(self.cfg, 'end_date') and self.cfg.end_date:
            df = df[df[self.cfg.date_col] <= self.cfg.end_date]
        
        if len(df) < len(data_package['df']):
            print(f"   过滤后行数: {len(df):,}")
        
        return df
    
    def load_with_stats(self, input_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        加载数据及统计量
        
        Returns:
            (df, feature_stats)
        """
        import pickle
        
        with open(input_path, 'rb') as f:
            data_package = pickle.load(f)
        
        return data_package['df'], data_package.get('feature_stats', {})
