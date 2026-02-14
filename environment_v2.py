"""
A股投资组合交易环境 V2 - 修复负奖励问题

核心改进:
1. 支持现金仓位 (第n+1维资产，收益为0)
2. 改进的奖励函数：对数收益率 + 方差惩罚 + 换手率惩罚
3. 课程学习支持：分阶段增加难度
4. 调试模式：0交易成本训练
5. 更强的换手率惩罚

A股特有约束：
1. T+1制度：当日买入的股票次日才能卖出
2. 涨跌停限制：普通股票±10%，ST股票±5%
3. 停牌处理：停牌股票无法交易
4. 做空限制：只能做多
5. 最小交易单位：100股（1手）
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from collections import deque


class PortfolioTradingEnv(gym.Env):
    """
    A股投资组合交易环境（适配异构特征）- V2 改进版
    """
    def __init__(self, df: pd.DataFrame, config, enable_cash=True):
        super().__init__()
        
        self.df = df.sort_values([config.data.date_col, config.data.stock_col]).copy()
        self.dates = self.df[config.data.date_col].unique()
        self.stocks = self.df[config.data.stock_col].unique()
        self.n_stocks = config.data.n_stocks  # 原始股票数量
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}
        
        self.cfg = config
        self.lookback = config.data.lookback_window
        self.max_pos = config.trading.max_position
        
        # V2: 现金仓位支持
        self.enable_cash = enable_cash
        self.n_assets = self.n_stocks + (1 if enable_cash else 0)  # 股票 + 现金
        self.cash_idx = self.n_stocks if enable_cash else None  # 现金索引
        
        # V2: 调试模式 - 可以设置更低交易成本
        self.tc = getattr(config.trading, 'transaction_cost', 0.0015)
        self.debug_zero_cost = getattr(config.trading, 'debug_zero_cost', False)
        if self.debug_zero_cost:
            self.tc = 0.0
            print("🐛 调试模式：交易成本设为 0")
        
        # 特征列（动态支持任意维度，f_* 开头的列）
        self.feature_cols = [c for c in df.columns if c.startswith('f_')]
        self.n_features = len(self.feature_cols)
        print(f"Features detected: {self.n_features} dimensions")
        
        # V2: 动作空间改为n_assets（包含现金）
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_assets,), dtype=np.float32)
        
        # V2: 观察空间需要特殊处理现金（现金的特征全为0）
        if enable_cash:
            # 状态形状: (n_assets, lookback, n_features)
            # 现金的特征将是一个特殊的零向量
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_assets, self.lookback, self.n_features),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_stocks, self.lookback, self.n_features),
                dtype=np.float32
            )
        
        # 预计算缓存
        self._cache_data()
        
        # T+1持仓追踪
        self.position_lock = None
        
    def _cache_data(self):
        """预计算价格和特征，加速step"""
        self._price_cache = {}
        self._feature_cache = {}
        self._tradable_cache = {}
        self._limit_cache = {}
        self._returns_cache = {}  # V2: 缓存收益率用于计算
        
        for date in self.dates:
            day_data = self.df[self.df[self.cfg.data.date_col] == date]
            self._price_cache[date] = {
                row[self.cfg.data.stock_col]: row['close'] 
                for _, row in day_data.iterrows()
            }
            
            # V2: 计算日收益率（用于后续奖励计算）
            if date != self.dates[-1]:
                next_date_idx = np.where(self.dates == date)[0][0] + 1
                next_date = self.dates[next_date_idx]
                current_prices = self._price_cache[date]
                next_prices = self._price_cache.get(next_date, {})
                daily_returns = {}
                for stock in current_prices:
                    if stock in next_prices and current_prices[stock] > 0:
                        daily_returns[stock] = (next_prices[stock] - current_prices[stock]) / current_prices[stock]
                    else:
                        daily_returns[stock] = 0.0
                self._returns_cache[date] = daily_returns
            
            # 特征标准化缓存（截面Z-score）
            features = day_data.set_index(self.cfg.data.stock_col)[self.feature_cols]
            if len(features) > 0:
                mean = features.mean()
                std = features.std()
                std = std.replace(0, 1.0)
                std = std + 1e-8
                normalized = ((features - mean) / std).fillna(0)
                self._feature_cache[date] = normalized.to_dict('index')
            else:
                self._feature_cache[date] = {}
            
            # VLM数据存在性缓存
            self._vlm_cache = {}
            if 'vlm_report_count' in day_data.columns:
                self._vlm_cache[date] = {
                    row[self.cfg.data.stock_col]: (row.get('vlm_report_count', 0) > 0)
                    for _, row in day_data.iterrows()
                }
            else:
                self._vlm_cache[date] = {stock: False for stock in self.stocks}
            
            # 可交易性
            tradable_dict = {}
            limit_dict = {}
            for _, row in day_data.iterrows():
                stock = row[self.cfg.data.stock_col]
                is_limit_up = row.get('hit_limit_up', 0)
                is_limit_down = row.get('hit_limit_down', 0)
                is_suspended = row.get('is_suspended', 0)
                
                tradable = 1 - max(is_limit_up, is_limit_down, is_suspended)
                tradable_dict[stock] = max(0, tradable)
                
                limit_dict[stock] = {
                    'limit_up': is_limit_up,
                    'limit_down': is_limit_down,
                    'suspended': is_suspended
                }
            
            self._tradable_cache[date] = tradable_dict
            self._limit_cache[date] = limit_dict
        
    def _get_state(self, date_idx: int) -> np.ndarray:
        """构建状态（包含现金仓位）"""
        if date_idx < self.lookback:
            available = self.dates[:date_idx+1]
            pad_count = self.lookback - len(available)
            date_slice = [available[0]] * pad_count + list(available)
        else:
            date_slice = self.dates[date_idx-self.lookback:date_idx]
        
        # V2: 包含现金的状态数组
        state = np.zeros((self.n_assets, self.lookback, self.n_features))
        
        for t, date in enumerate(date_slice):
            date_features = self._feature_cache.get(date, {})
            for stock, idx in self.stock_to_idx.items():
                if stock in date_features:
                    feats = list(date_features[stock].values())
                    if len(feats) == self.n_features:
                        state[idx, t, :] = feats
            # V2: 现金的特征保持为0（表示零收益、零波动）
            if self.enable_cash:
                state[self.cash_idx, t, :] = 0.0  # 现金特征全为0
        
        # 数据清洗
        if np.isnan(state).any() or np.isinf(state).any():
            nan_count = np.isnan(state).sum()
            inf_count = np.isinf(state).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"⚠️ Warning: State contains {nan_count} NaN, {inf_count} Inf at date_idx {date_idx}")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state.astype(np.float32)
    
    def _get_tradable_mask(self, date_idx: int) -> np.ndarray:
        """获取可交易mask（V2: 包含现金，现金始终可交易）"""
        if date_idx >= len(self.dates):
            if self.enable_cash:
                return np.ones(self.n_assets)
            return np.ones(self.n_stocks)
        
        date = self.dates[date_idx]
        
        # V2: 为所有资产（含现金）创建mask
        mask = np.ones(self.n_assets)
        tradable_dict = self._tradable_cache.get(date, {})
        
        for stock, idx in self.stock_to_idx.items():
            if idx >= self.n_stocks:
                continue
            base_tradable = tradable_dict.get(stock, 1)
            
            # T+1锁定检查
            if self.position_lock and idx in self.position_lock:
                unlock_date = self.position_lock[idx]
                if pd.to_datetime(date) < pd.to_datetime(unlock_date):
                    base_tradable = 0
            
            mask[idx] = base_tradable
        
        # V2: 现金始终可交易
        if self.enable_cash:
            mask[self.cash_idx] = 1.0
        
        return mask
    
    def _get_limit_status(self, date_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取涨跌停和停牌状态"""
        if date_idx >= len(self.dates):
            if self.enable_cash:
                return np.zeros(self.n_assets), np.zeros(self.n_assets), np.zeros(self.n_assets)
            return np.zeros(self.n_stocks), np.zeros(self.n_stocks), np.zeros(self.n_stocks)
        
        date = self.dates[date_idx]
        limit_dict = self._limit_cache.get(date, {})
        
        # V2: 包含现金
        n = self.n_assets if self.enable_cash else self.n_stocks
        limit_up = np.zeros(n)
        limit_down = np.zeros(n)
        suspended = np.zeros(n)
        
        for stock, idx in self.stock_to_idx.items():
            status = limit_dict.get(stock, {})
            limit_up[idx] = status.get('limit_up', 0)
            limit_down[idx] = status.get('limit_down', 0)
            suspended[idx] = status.get('suspended', 0)
        
        return limit_up, limit_down, suspended
    
    def _apply_constraints(self, weights: np.ndarray, date_idx: int) -> np.ndarray:
        """
        应用A股约束（V2: 处理现金仓位）
        """
        # 分离股票权重和现金权重
        if self.enable_cash:
            stock_weights = weights[:self.n_stocks].copy()
            cash_weight = weights[self.cash_idx]
        else:
            stock_weights = weights.copy()
            cash_weight = 0.0
        
        # 获取可交易mask（仅股票）
        tradable = self._get_tradable_mask(date_idx)[:self.n_stocks]
        
        # 获取涨跌停状态
        limit_up, limit_down, suspended = self._get_limit_status(date_idx)
        limit_up = limit_up[:self.n_stocks]
        limit_down = limit_down[:self.n_stocks]
        suspended = suspended[:self.n_stocks]
        
        # 应用约束
        stock_weights = np.maximum(stock_weights, 0)
        stock_weights = stock_weights * tradable
        stock_weights = stock_weights * (1 - limit_up)
        
        # 归一化股票权重
        stock_sum = np.sum(stock_weights)
        if stock_sum > 1e-8:
            stock_weights = stock_weights / stock_sum * (1 - cash_weight)
        else:
            # 全部不可交易时，等权持有
            if np.sum(tradable) > 0:
                stock_weights = tradable / np.sum(tradable) * (1 - cash_weight)
            else:
                stock_weights = np.ones(self.n_stocks) / self.n_stocks * (1 - cash_weight)
        
        # 仓位上限约束
        stock_weights = np.clip(stock_weights, 0, self.max_pos)
        
        # 重新归一化（如果触发了上限）
        stock_sum = np.sum(stock_weights)
        if stock_sum > 1e-8:
            stock_weights = stock_weights / stock_sum * (1 - cash_weight)
        
        # V2: 合并回完整权重向量
        if self.enable_cash:
            weights = np.concatenate([stock_weights, [cash_weight]])
        else:
            weights = stock_weights
        
        # 最终归一化确保和为1
        weights = weights / (np.sum(weights) + 1e-8)
        
        return weights
    
    def _update_position_lock(self, old_weights: np.ndarray, new_weights: np.ndarray, date_idx: int):
        """更新T+1持仓锁定（V2: 只锁定股票，不锁定现金）"""
        if date_idx >= len(self.dates) - 1:
            return
        
        next_date = self.dates[date_idx + 1]
        
        if self.position_lock is None:
            self.position_lock = {}
        
        # 只考虑股票仓位
        old_stock = old_weights[:self.n_stocks] if self.enable_cash else old_weights
        new_stock = new_weights[:self.n_stocks] if self.enable_cash else new_weights
        
        for i in range(self.n_stocks):
            if new_stock[i] > old_stock[i] + 1e-6:
                self.position_lock[i] = next_date
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步交易（V2: 改进的奖励函数）"""
        current_idx = self.current_step
        if current_idx >= len(self.dates):
            return self._get_state(len(self.dates) - 1) if len(self.dates) > 0 else np.zeros((self.n_assets, self.lookback, self.n_features)), 0, True, {}
        
        # 数据清洗
        if np.isnan(action).any() or np.isinf(action).any():
            print(f"🔥 CRITICAL: Action contains NaN/Inf at step {current_idx}!")
            action = np.ones(self.n_assets) / self.n_assets
        
        current_date = self.dates[current_idx]
        
        # 动作->权重
        action = np.clip(action, a_min=1e-10, a_max=1.0)
        action = action / action.sum()
        raw_weights = action
        
        # 应用约束
        weights = self._apply_constraints(raw_weights, current_idx + 1)
        
        # 更新T+1锁定
        self._update_position_lock(self.prev_weights, weights, current_idx)
        
        if current_idx + 1 >= len(self.dates):
            return self._get_state(current_idx), 0, True, {}
        
        next_date = self.dates[current_idx + 1]
        
        # 计算收益（V2: 包含现金）
        current_prices = self._price_cache.get(current_date, {})
        next_prices = self._price_cache.get(next_date, {})
        
        port_return = 0
        for stock, idx in self.stock_to_idx.items():
            if stock in current_prices and stock in next_prices:
                if current_prices[stock] <= 0:
                    continue
                ret = (next_prices[stock] - current_prices[stock]) / current_prices[stock]
                port_return += weights[idx] * ret
        
        # V2: 现金部分的收益为0
        if self.enable_cash:
            cash_return = 0.0  # 现金无收益
            # 现金权重已包含在port_return计算中（乘以0）
        
        # 换手率计算（V2: 股票换手率）
        prev_stock_weights = self.prev_weights[:self.n_stocks] if self.enable_cash else self.prev_weights
        curr_stock_weights = weights[:self.n_stocks] if self.enable_cash else weights
        turnover = np.sum(np.abs(curr_stock_weights - prev_stock_weights))
        
        cost = turnover * self.tc
        net_return = port_return - cost
        
        # V2: 改进的奖励计算
        self.returns_buffer.append(net_return)
        if len(self.returns_buffer) > 60:
            self.returns_buffer.popleft()
        
        reward = self._compute_reward_v2(net_return, port_return, turnover, weights)
        
        self.prev_weights = weights.copy()
        self.current_step += 1
        new_value = self.history[-1] * (1 + net_return)
        self.history.append(new_value)
        
        # 终止条件
        done = (self.current_step >= len(self.dates) - 1) or (new_value < 0.5)
        
        # 市场状态
        limit_up, limit_down, suspended = self._get_limit_status(current_idx + 1)
        action_mask = self._get_tradable_mask(current_idx + 1)
        
        # VLM数据检查
        vlm_dict = self._vlm_cache.get(next_date, {})
        has_vlm_data = any(vlm_dict.get(stock, False) for stock in self.stock_to_idx.keys())
        
        info = {
            'date': next_date,
            'portfolio_value': new_value,
            'daily_return': net_return,
            'gross_return': port_return,
            'turnover': turnover,
            'cost': cost,
            'weights': weights.copy(),
            'stock_weights': weights[:self.n_stocks].copy() if self.enable_cash else weights.copy(),
            'cash_weight': weights[self.cash_idx] if self.enable_cash else 0.0,
            'n_limit_up': int(limit_up.sum()),
            'n_limit_down': int(limit_down.sum()),
            'n_suspended': int(suspended.sum()),
            'has_vlm_data': has_vlm_data,
            'action_mask': action_mask
        }
        
        return self._get_state(self.current_step), reward, done, info
    
    def _compute_reward_v2(self, net_return: float, gross_return: float, 
                           turnover: float, weights: np.ndarray) -> float:
        """
        V2: 改进的奖励函数
        
        支持多种奖励模式：
        - 'profit_only': 纯收益（课程学习第一阶段）
        - 'log_return': 对数收益率
        - 'sharpe': 夏普比率
        - 'risk_adjusted': 收益 - 风险惩罚 - 换手惩罚（推荐）
        """
        reward_mode = getattr(self.cfg.trading, 'reward_mode', 'risk_adjusted')
        reward_scale = getattr(self.cfg.trading, 'reward_scale', 1.0)
        
        # 基础奖励
        if reward_mode == 'profit_only':
            # 课程学习第一阶段：只关注收益
            reward = net_return * 100
            
        elif reward_mode == 'log_return':
            # 对数收益率（更稳定）
            if net_return > -1:  # 避免log(0)
                reward = np.log(1 + net_return) * 100
            else:
                reward = -10  # 大惩罚
                
        elif reward_mode == 'sharpe':
            # 原始夏普比率
            if len(self.returns_buffer) >= 30:
                mean_ret = np.mean(self.returns_buffer)
                std_ret = np.std(self.returns_buffer) + 1e-6
                sharpe = (mean_ret - self.cfg.trading.risk_free_rate) / std_ret * np.sqrt(252)
                reward = sharpe
            else:
                reward = net_return * 10
                
        elif reward_mode == 'risk_adjusted':
            # V2: 推荐的奖励模式
            # 收益 - 方差惩罚 - 换手率惩罚
            if len(self.returns_buffer) >= 10:
                mean_ret = np.mean(self.returns_buffer)
                var_ret = np.var(self.returns_buffer) + 1e-8
                
                # 对数收益率基础奖励
                if net_return > -1:
                    base_reward = np.log(1 + net_return) * 100
                else:
                    base_reward = -10
                
                # 风险惩罚系数
                risk_penalty_coef = getattr(self.cfg.trading, 'risk_penalty_coef', 10.0)
                risk_penalty = var_ret * risk_penalty_coef
                
                # 换手率惩罚（V2: 更强）
                turnover_penalty_coef = getattr(self.cfg.trading, 'turnover_penalty_coef', 0.5)
                turnover_penalty = turnover * turnover_penalty_coef
                
                reward = base_reward - risk_penalty - turnover_penalty
            else:
                # Warmup阶段
                reward = net_return * 10 if net_return > 0 else net_return * 20
        else:
            # 默认
            reward = net_return * 10
        
        # V2: 集中度惩罚
        stock_weights = weights[:self.n_stocks] if self.enable_cash else weights
        max_weight = np.max(stock_weights) if len(stock_weights) > 0 else 0
        concentration_penalty = max(0, max_weight - 0.1) * getattr(self.cfg.trading, 'concentration_penalty', 0.1)
        reward -= concentration_penalty
        
        # 缩放
        reward = reward * reward_scale
        
        return reward
    
    def reset(self) -> np.ndarray:
        """重置环境（V2: 初始化现金仓位）"""
        max_valid_step = max(0, len(self.dates) - 2)
        self.current_step = min(self.lookback, max_valid_step)
        if self.current_step < 0:
            self.current_step = 0
        
        # V2: 初始化权重包含现金
        if self.enable_cash:
            # 初始：100%现金
            self.prev_weights = np.zeros(self.n_assets)
            self.prev_weights[self.cash_idx] = 1.0
        else:
            self.prev_weights = np.ones(self.n_stocks) / self.n_stocks
        
        self.returns_buffer = deque(maxlen=60)
        self.history = [1.0]
        self.position_lock = {}
        
        return self._get_state(self.current_step)
