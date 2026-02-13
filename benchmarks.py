"""
对比策略集合 (Benchmark Strategies)

严格公平对比原则 (方案A):
- 相同股票池: 471只A股
- 相同交易成本: 0.15% 双边
- 相同约束: 单股10%上限, 无做空
- 相同回测频率: 日度再平衡
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, name: str, df: pd.DataFrame, config):
        self.name = name
        self.df = df.sort_values([config.data.date_col, config.data.stock_col]).copy()
        self.dates = self.df[config.data.date_col].unique()
        self.stocks = self.df[config.data.stock_col].unique()
        self.n_stocks = len(self.stocks)
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}
        self.cfg = config
        
        # 缓存数据
        self._cache_data()
        
    def _cache_data(self):
        """预计算价格和特征"""
        self._price_cache = {}
        self._return_cache = {}
        self._feature_cache = {}
        self._tradable_cache = {}
        
        for date in self.dates:
            day_data = self.df[self.df[self.cfg.data.date_col] == date]
            
            # 价格
            self._price_cache[date] = {
                row[self.cfg.data.stock_col]: row['close'] 
                for _, row in day_data.iterrows()
            }
            
            # 日收益率
            self._return_cache[date] = {
                row[self.cfg.data.stock_col]: row.get('daily_return', 0)
                for _, row in day_data.iterrows()
            }
            
            # 可交易性
            self._tradable_cache[date] = {
                row[self.cfg.data.stock_col]: row.get('tradable', 1)
                for _, row in day_data.iterrows()
            }
            
            # 特征
            feature_cols = [c for c in day_data.columns if c.startswith('f_')]
            if feature_cols:
                self._feature_cache[date] = day_data.set_index(
                    self.cfg.data.stock_col
                )[feature_cols].to_dict('index')
    
    def _get_tradable_mask(self, date_idx: int) -> np.ndarray:
        """获取可交易mask"""
        if date_idx >= len(self.dates):
            return np.ones(self.n_stocks)
        date = self.dates[date_idx]
        mask = np.ones(self.n_stocks)
        tradable_dict = self._tradable_cache.get(date, {})
        for stock, idx in self.stock_to_idx.items():
            mask[idx] = tradable_dict.get(stock, 1)
        return mask
    
    def _apply_constraints(self, weights: np.ndarray, date_idx: int) -> np.ndarray:
        """应用约束: 非负、仓位上限、归一化"""
        tradable = self._get_tradable_mask(date_idx)
        
        # 非负约束（无做空）
        weights = np.maximum(weights, 0)
        
        # 可交易mask
        weights = weights * tradable
        
        # 归一化
        if np.sum(weights) > 1e-8:
            weights = weights / np.sum(weights)
        else:
            weights = tradable / np.sum(tradable) if np.sum(tradable) > 0 else \
                     np.ones(self.n_stocks) / self.n_stocks
        
        # 仓位上限
        weights = np.clip(weights, 0, self.cfg.trading.max_position)
        weights = weights / np.sum(weights)
        
        return weights
    
    @abstractmethod
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """
        计算目标权重
        
        Args:
            date_idx: 当前日期索引
            current_weights: 当前持仓权重
            
        Returns:
            目标权重 (n_stocks,)
        """
        pass
    
    def backtest(self) -> Dict:
        """运行回测"""
        portfolio_value = 1.0
        current_weights = np.ones(self.n_stocks) / self.n_stocks
        history = [portfolio_value]
        returns_history = []
        turnover_history = []
        weights_history = []
        
        for i in range(len(self.dates) - 1):
            current_date = self.dates[i]
            next_date = self.dates[i + 1]
            
            # 获取目标权重
            target_weights = self.get_weights(i, current_weights)
            target_weights = self._apply_constraints(target_weights, i)
            
            # 计算换手率
            turnover = np.sum(np.abs(target_weights - current_weights))
            turnover_history.append(turnover)
            
            # 计算组合收益
            current_prices = self._price_cache.get(current_date, {})
            next_prices = self._price_cache.get(next_date, {})
            
            port_return = 0
            for stock, idx in self.stock_to_idx.items():
                if stock in current_prices and stock in next_prices:
                    ret = (next_prices[stock] - current_prices[stock]) / current_prices[stock]
                    port_return += target_weights[idx] * ret
            
            # 扣除交易成本
            cost = turnover * self.cfg.trading.transaction_cost
            net_return = port_return - cost
            
            portfolio_value *= (1 + net_return)
            history.append(portfolio_value)
            returns_history.append(net_return)
            weights_history.append(target_weights.copy())
            
            # 更新当前权重（假设再平衡后持有到次日）
            current_weights = target_weights.copy()
        
        # 计算指标
        returns = np.array(returns_history)
        
        metrics = {
            'name': self.name,
            'total_return': (portfolio_value - 1) * 100,
            'annual_return': ((portfolio_value ** (252/len(returns))) - 1) * 100,
            'sharpe': self._calc_sharpe(returns),
            'max_drawdown': self._calc_max_drawdown(history),
            'volatility': np.std(returns) * np.sqrt(252) * 100,
            'win_rate': np.mean(returns > 0) * 100,
            'avg_turnover': np.mean(turnover_history) * 100,
            'history': history,
            'returns': returns,
            'weights_history': weights_history
        }
        
        return metrics
    
    def _calc_sharpe(self, returns: np.ndarray) -> float:
        """计算Sharpe比率"""
        if len(returns) < 30:
            return 0.0
        excess = returns - self.cfg.trading.risk_free_rate
        std = np.std(excess)
        if std < 1e-8:
            return 0.0
        return np.mean(excess) / std * np.sqrt(252)
    
    def _calc_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        values = np.array(values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown) * 100


class CSI300Benchmark(BaseStrategy):
    """
    沪深300指数基准
    
    使用CSI300的历史收益率作为基准
    假设无法完全复制指数，使用跟踪误差
    """
    def __init__(self, df: pd.DataFrame, config, csi300_returns: pd.Series = None):
        super().__init__("CSI300_Benchmark", df, config)
        self.csi300_returns = csi300_returns
        
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """
        沪深300指数策略：直接按指数收益率调整组合
        简化处理：等权持有（实际应用需要CSI300权重数据）
        """
        # 如果没有CSI300数据，使用等权作为近似
        if self.csi300_returns is None:
            return np.ones(self.n_stocks) / self.n_stocks
        
        # 实际应使用CSI300成分股权重
        # 这里简化为等权
        return np.ones(self.n_stocks) / self.n_stocks


class EqualWeightStrategy(BaseStrategy):
    """
    等权重策略
    
    每日等权持有所有可交易股票
    消除选股能力，检验择时价值
    """
    def __init__(self, df: pd.DataFrame, config):
        super().__init__("Equal_Weight", df, config)
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """等权重"""
        tradable = self._get_tradable_mask(date_idx)
        weights = tradable / np.sum(tradable) if np.sum(tradable) > 0 else \
                  np.ones(self.n_stocks) / self.n_stocks
        return weights


class BuyHoldStrategy(BaseStrategy):
    """
    买入持有策略
    
    期初等权买入，此后不再调仓
    检验交易频率的价值
    """
    def __init__(self, df: pd.DataFrame, config):
        super().__init__("Buy_Hold", df, config)
        self.initial_weights = None
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """买入持有：期初确定权重后不变"""
        if self.initial_weights is None:
            tradable = self._get_tradable_mask(date_idx)
            self.initial_weights = tradable / np.sum(tradable) if np.sum(tradable) > 0 else \
                                   np.ones(self.n_stocks) / self.n_stocks
        return self.initial_weights


class MomentumStrategy(BaseStrategy):
    """
    动量策略 (20日)
    
    按20日收益率排序，做多Top 10%股票
    检验动量因子有效性
    """
    def __init__(self, df: pd.DataFrame, config, lookback: int = 20, top_pct: float = 0.1):
        super().__init__("Momentum_20D", df, config)
        self.lookback = lookback
        self.top_pct = top_pct
        self._precompute_momentum()
    
    def _precompute_momentum(self):
        """预计算动量"""
        self.momentum_cache = {}
        
        for i, date in enumerate(self.dates):
            if i < self.lookback:
                self.momentum_cache[date] = {}
                continue
            
            # 获取过去lookback天的价格
            start_date = self.dates[i - self.lookback]
            start_prices = self._price_cache.get(start_date, {})
            end_prices = self._price_cache.get(date, {})
            
            momentum = {}
            for stock in self.stocks:
                if stock in start_prices and stock in end_prices and start_prices[stock] > 0:
                    mom = (end_prices[stock] - start_prices[stock]) / start_prices[stock]
                    momentum[stock] = mom
                else:
                    momentum[stock] = -np.inf  # 数据缺失排最后
            
            self.momentum_cache[date] = momentum
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """动量：做多过去20日收益率最高的股票"""
        if date_idx < self.lookback:
            # 前lookback天等权
            tradable = self._get_tradable_mask(date_idx)
            return tradable / np.sum(tradable)
        
        date = self.dates[date_idx]
        momentum = self.momentum_cache.get(date, {})
        tradable = self._get_tradable_mask(date_idx)
        
        # 筛选可交易股票
        tradable_stocks = [s for s in self.stocks if tradable[self.stock_to_idx[s]] > 0]
        
        if len(tradable_stocks) == 0:
            return np.ones(self.n_stocks) / self.n_stocks
        
        # 按动量排序
        sorted_stocks = sorted(tradable_stocks, key=lambda s: momentum.get(s, -np.inf), reverse=True)
        
        # 选择Top N
        n_select = max(1, int(len(tradable_stocks) * self.top_pct))
        selected = set(sorted_stocks[:n_select])
        
        # 等权配置
        weights = np.zeros(self.n_stocks)
        for stock in selected:
            weights[self.stock_to_idx[stock]] = 1.0
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else \
                  np.ones(self.n_stocks) / self.n_stocks
        
        return weights


class MeanReversionStrategy(BaseStrategy):
    """
    反转策略 (20日)
    
    按20日收益率排序，做多Bottom 10%股票（跌得最多的）
    检验反转因子有效性
    """
    def __init__(self, df: pd.DataFrame, config, lookback: int = 20, bottom_pct: float = 0.1):
        super().__init__("Mean_Reversion", df, config)
        self.lookback = lookback
        self.bottom_pct = bottom_pct
        self._precompute_returns()
    
    def _precompute_returns(self):
        """预计算收益率"""
        self.return_cache = {}
        
        for i, date in enumerate(self.dates):
            if i < self.lookback:
                self.return_cache[date] = {}
                continue
            
            start_date = self.dates[i - self.lookback]
            start_prices = self._price_cache.get(start_date, {})
            end_prices = self._price_cache.get(date, {})
            
            ret_dict = {}
            for stock in self.stocks:
                if stock in start_prices and stock in end_prices and start_prices[stock] > 0:
                    ret = (end_prices[stock] - start_prices[stock]) / start_prices[stock]
                    ret_dict[stock] = ret
                else:
                    ret_dict[stock] = np.inf  # 数据缺失排最后（反转选小的）
            
            self.return_cache[date] = ret_dict
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """反转：做多过去20日收益率最低的股票"""
        if date_idx < self.lookback:
            tradable = self._get_tradable_mask(date_idx)
            return tradable / np.sum(tradable)
        
        date = self.dates[date_idx]
        returns = self.return_cache.get(date, {})
        tradable = self._get_tradable_mask(date_idx)
        
        tradable_stocks = [s for s in self.stocks if tradable[self.stock_to_idx[s]] > 0]
        
        if len(tradable_stocks) == 0:
            return np.ones(self.n_stocks) / self.n_stocks
        
        # 按收益率排序（选最低的）
        sorted_stocks = sorted(tradable_stocks, key=lambda s: returns.get(s, np.inf))
        
        n_select = max(1, int(len(tradable_stocks) * self.bottom_pct))
        selected = set(sorted_stocks[:n_select])
        
        weights = np.zeros(self.n_stocks)
        for stock in selected:
            weights[self.stock_to_idx[stock]] = 1.0
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else \
                  np.ones(self.n_stocks) / self.n_stocks
        
        return weights


class LowVolatilityStrategy(BaseStrategy):
    """
    低波动策略
    
    按20日波动率排序，做多波动率最低的10%股票
    检验低波动异象
    """
    def __init__(self, df: pd.DataFrame, config, lookback: int = 20, bottom_pct: float = 0.1):
        super().__init__("Low_Volatility", df, config)
        self.lookback = lookback
        self.bottom_pct = bottom_pct
        self._precompute_volatility()
    
    def _precompute_volatility(self):
        """预计算波动率"""
        self.volatility_cache = {}
        
        for i, date in enumerate(self.dates):
            if i < self.lookback:
                self.volatility_cache[date] = {}
                continue
            
            # 获取过去lookback天的收益率
            returns_list = []
            for j in range(i - self.lookback + 1, i + 1):
                d = self.dates[j]
                day_returns = self._return_cache.get(d, {})
                returns_list.append(day_returns)
            
            # 计算每只股票的标准差
            vol_dict = {}
            for stock in self.stocks:
                stock_returns = []
                for day_ret in returns_list:
                    if stock in day_ret and not np.isnan(day_ret[stock]):
                        stock_returns.append(day_ret[stock])
                
                if len(stock_returns) >= self.lookback * 0.5:  # 至少50%数据
                    vol_dict[stock] = np.std(stock_returns)
                else:
                    vol_dict[stock] = np.inf  # 数据不足排最后
            
            self.volatility_cache[date] = vol_dict
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """低波动：做多波动率最低的股票"""
        if date_idx < self.lookback:
            tradable = self._get_tradable_mask(date_idx)
            return tradable / np.sum(tradable)
        
        date = self.dates[date_idx]
        volatility = self.volatility_cache.get(date, {})
        tradable = self._get_tradable_mask(date_idx)
        
        tradable_stocks = [s for s in self.stocks if tradable[self.stock_to_idx[s]] > 0]
        
        if len(tradable_stocks) == 0:
            return np.ones(self.n_stocks) / self.n_stocks
        
        # 按波动率排序（选最低的）
        sorted_stocks = sorted(tradable_stocks, key=lambda s: volatility.get(s, np.inf))
        
        n_select = max(1, int(len(tradable_stocks) * self.bottom_pct))
        selected = set(sorted_stocks[:n_select])
        
        weights = np.zeros(self.n_stocks)
        for stock in selected:
            weights[self.stock_to_idx[stock]] = 1.0
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else \
                  np.ones(self.n_stocks) / self.n_stocks
        
        return weights


class SentimentDrivenStrategy(BaseStrategy):
    """
    情绪驱动策略
    
    综合股吧bullishness和VLM sentiment_score
    做多情绪指标最高的10%股票
    检验多模态情绪数据的价值
    """
    def __init__(self, df: pd.DataFrame, config, top_pct: float = 0.1):
        super().__init__("Sentiment_Driven", df, config)
        self.top_pct = top_pct
        self._precompute_sentiment()
    
    def _precompute_sentiment(self):
        """预计算综合情绪得分"""
        self.sentiment_cache = {}
        
        for date in self.dates:
            day_data = self.df[self.df[self.cfg.data.date_col] == date]
            sentiment = {}
            
            for _, row in day_data.iterrows():
                stock = row[self.cfg.data.stock_col]
                
                # 获取股吧情绪
                guba_bullishness = 0
                for col in ['guba_bullishness', 'f_bullishness', 'f_guba_bullishness']:
                    if col in row and not pd.isna(row[col]):
                        guba_bullishness = row[col]
                        break
                
                # 获取VLM情绪
                vlm_sentiment = 0
                for col in ['vlm_sentiment_score', 'f_vlm_sentiment_score', 'sentiment_score']:
                    if col in row and not pd.isna(row[col]):
                        vlm_sentiment = row[col]
                        break
                
                # 综合情绪得分（标准化后加权）
                # 股吧情绪通常在0-10或0-100，VLM在0-100
                guba_normalized = guba_bullishness / 10 if guba_bullishness > 10 else guba_bullishness
                vlm_normalized = vlm_sentiment / 100
                
                # 加权平均：VLM 60%，股吧 40%
                combined = 0.6 * vlm_normalized + 0.4 * guba_normalized
                sentiment[stock] = combined
            
            self.sentiment_cache[date] = sentiment
    
    def get_weights(self, date_idx: int, current_weights: np.ndarray) -> np.ndarray:
        """情绪驱动：做多情绪得分最高的股票"""
        date = self.dates[date_idx]
        sentiment = self.sentiment_cache.get(date, {})
        tradable = self._get_tradable_mask(date_idx)
        
        # 如果没有情绪数据，使用等权
        if not sentiment:
            return tradable / np.sum(tradable) if np.sum(tradable) > 0 else \
                   np.ones(self.n_stocks) / self.n_stocks
        
        tradable_stocks = [s for s in self.stocks if tradable[self.stock_to_idx[s]] > 0]
        
        if len(tradable_stocks) == 0:
            return np.ones(self.n_stocks) / self.n_stocks
        
        # 按情绪排序
        sorted_stocks = sorted(tradable_stocks, key=lambda s: sentiment.get(s, -np.inf), reverse=True)
        
        n_select = max(1, int(len(tradable_stocks) * self.top_pct))
        selected = set(sorted_stocks[:n_select])
        
        weights = np.zeros(self.n_stocks)
        for stock in selected:
            weights[self.stock_to_idx[stock]] = 1.0
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else \
                  np.ones(self.n_stocks) / self.n_stocks
        
        return weights


def run_all_benchmarks(df: pd.DataFrame, config, verbose: bool = True) -> Dict[str, Dict]:
    """
    运行所有对比策略
    
    Args:
        df: 融合后的特征数据
        config: 配置对象
        verbose: 是否打印详细信息
        
    Returns:
        策略名 -> 指标字典 的映射
    """
    strategies = [
        CSI300Benchmark(df, config),
        EqualWeightStrategy(df, config),
        BuyHoldStrategy(df, config),
        MomentumStrategy(df, config, lookback=20, top_pct=0.1),
        MeanReversionStrategy(df, config, lookback=20, bottom_pct=0.1),
        LowVolatilityStrategy(df, config, lookback=20, bottom_pct=0.1),
        SentimentDrivenStrategy(df, config, top_pct=0.1),
    ]
    
    results = {}
    
    if verbose:
        print("\n" + "="*80)
        print("Running Benchmark Strategies")
        print("="*80)
    
    for strategy in strategies:
        if verbose:
            print(f"\nRunning {strategy.name}...")
        
        result = strategy.backtest()
        results[strategy.name] = result
        
        if verbose:
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Sharpe: {result['sharpe']:.2f}")
            print(f"  Max DD: {result['max_drawdown']:.2f}%")
            print(f"  Turnover: {result['avg_turnover']:.2f}%")
    
    if verbose:
        print("\n" + "="*80)
        print("Benchmark Summary")
        print("="*80)
        print(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>10} {'MaxDD':>10} {'Turnover':>10}")
        print("-"*80)
        for name, result in results.items():
            print(f"{name:<20} {result['total_return']:>9.2f}% {result['sharpe']:>10.2f} "
                  f"{result['max_drawdown']:>9.2f}% {result['avg_turnover']:>9.2f}%")
    
    return results
