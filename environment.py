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
    A股投资组合交易环境（适配异构40维特征）
    
    A股特有约束：
    1. T+1制度：当日买入的股票次日才能卖出
    2. 涨跌停限制：普通股票±10%，ST股票±5%
    3. 停牌处理：停牌股票无法交易
    4. 做空限制：只能做多
    5. 最小交易单位：100股（1手）
    """
    def __init__(self, df: pd.DataFrame, config):
        super().__init__()
        
        self.df = df.sort_values([config.data.date_col, config.data.stock_col]).copy()
        self.dates = self.df[config.data.date_col].unique()
        self.stocks = self.df[config.data.stock_col].unique()
        self.n_stocks = config.data.n_stocks  # 使用全局配置，确保与 agent 网络输出维度一致
        self.stock_to_idx = {s: i for i, s in enumerate(self.stocks)}
        
        self.cfg = config
        self.lookback = config.data.lookback_window
        self.tc = config.trading.transaction_cost
        self.max_pos = config.trading.max_position
        
        # 特征列（动态支持任意维度，f_* 开头的列）
        self.feature_cols = [c for c in df.columns if c.startswith('f_')]
        self.n_features = len(self.feature_cols)
        print(f"Features detected: {self.n_features} dimensions")
        # 移除硬性40维要求，支持动态特征数
        
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_stocks,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks, self.lookback, self.n_features),
            dtype=np.float32
        )
        
        # 预计算缓存
        self._cache_data()
        
        # T+1持仓追踪：记录每只股票的持仓成本日和可卖日期
        self.position_lock = None  # {stock_idx: unlock_date}
        
    def _cache_data(self):
        """预计算价格和特征，加速step"""
        self._price_cache = {}
        self._feature_cache = {}
        self._tradable_cache = {}
        self._limit_cache = {}  # 涨跌停缓存
        
        for date in self.dates:
            day_data = self.df[self.df[self.cfg.data.date_col] == date]
            self._price_cache[date] = {
                row[self.cfg.data.stock_col]: row['close'] 
                for _, row in day_data.iterrows()
            }
            
            # 特征标准化缓存（截面Z-score）
            features = day_data.set_index(self.cfg.data.stock_col)[self.feature_cols]
            if len(features) > 0:
                mean = features.mean()
                std = features.std()
                # 处理 std=0 的情况（所有股票该特征值相同）
                std = std.replace(0, 1.0)  # 如果std=0，设为1避免除以0
                std = std + 1e-8  # 再加小量保险
                
                # 标准化并处理可能的 NaN
                normalized = ((features - mean) / std)
                normalized = normalized.fillna(0)  # 填充任何产生的NaN
                
                self._feature_cache[date] = normalized.to_dict('index')
            else:
                self._feature_cache[date] = {}
            
            # VLM数据存在性缓存（用于优先采样）
            self._vlm_cache = {}
            if 'vlm_report_count' in day_data.columns:
                self._vlm_cache[date] = {
                    row[self.cfg.data.stock_col]: (row.get('vlm_report_count', 0) > 0)
                    for _, row in day_data.iterrows()
                }
            else:
                self._vlm_cache[date] = {stock: False for stock in self.stocks}
            
            # 可交易性（综合涨跌停和停牌）
            tradable_dict = {}
            limit_dict = {}
            for _, row in day_data.iterrows():
                stock = row[self.cfg.data.stock_col]
                # 可交易条件：非涨停、非跌停、非停牌
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
        """构建状态（过去lookback天40维特征）"""
        if date_idx < self.lookback:
            available = self.dates[:date_idx+1]
            pad_count = self.lookback - len(available)
            date_slice = [available[0]] * pad_count + list(available)
        else:
            date_slice = self.dates[date_idx-self.lookback:date_idx]
        
        state = np.zeros((self.n_stocks, self.lookback, self.n_features))
        
        for t, date in enumerate(date_slice):
            date_features = self._feature_cache.get(date, {})
            for stock, idx in self.stock_to_idx.items():
                if stock in date_features:
                    feats = list(date_features[stock].values())
                    if len(feats) == self.n_features:
                        state[idx, t, :] = feats
        
        # 数据清洗：检查并修复 NaN/Inf
        if np.isnan(state).any() or np.isinf(state).any():
            nan_count = np.isnan(state).sum()
            inf_count = np.isinf(state).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"⚠️ Warning: State contains {nan_count} NaN, {inf_count} Inf at date_idx {date_idx}")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state.astype(np.float32)
    
    def _get_tradable_mask(self, date_idx: int) -> np.ndarray:
        """获取可交易mask（考虑T+1锁定）"""
        if date_idx >= len(self.dates):
            return np.ones(self.n_stocks)
        
        date = self.dates[date_idx]
        mask = np.ones(self.n_stocks)
        tradable_dict = self._tradable_cache.get(date, {})
        
        for stock, idx in self.stock_to_idx.items():
            # 确保索引在有效范围内
            if idx >= self.n_stocks:
                continue
            base_tradable = tradable_dict.get(stock, 1)
            
            # T+1锁定检查
            if self.position_lock and idx in self.position_lock:
                unlock_date = self.position_lock[idx]
                if pd.to_datetime(date) < pd.to_datetime(unlock_date):
                    # 还不能卖，设为0（但如果原本就不可交易也不改变）
                    base_tradable = 0
            
            mask[idx] = base_tradable
        
        return mask
    
    def _get_limit_status(self, date_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取涨跌停和停牌状态"""
        if date_idx >= len(self.dates):
            return np.zeros(self.n_stocks), np.zeros(self.n_stocks), np.zeros(self.n_stocks)
        
        date = self.dates[date_idx]
        limit_dict = self._limit_cache.get(date, {})
        
        limit_up = np.zeros(self.n_stocks)
        limit_down = np.zeros(self.n_stocks)
        suspended = np.zeros(self.n_stocks)
        
        for stock, idx in self.stock_to_idx.items():
            status = limit_dict.get(stock, {})
            limit_up[idx] = status.get('limit_up', 0)
            limit_down[idx] = status.get('limit_down', 0)
            suspended[idx] = status.get('suspended', 0)
        
        return limit_up, limit_down, suspended
    
    def _apply_constraints(self, weights: np.ndarray, date_idx: int) -> np.ndarray:
        """
        应用A股约束：
        1. 涨跌停mask（涨停不能买，跌停不能卖）
        2. T+1锁定（当日买入次日才能卖）
        3. 仓位上限
        4. 做空限制（非负权重）
        """
        # 获取可交易mask
        tradable = self._get_tradable_mask(date_idx)
        # 确保形状匹配（防御性编程）
        if tradable.shape[0] != self.n_stocks:
            tradable = np.ones(self.n_stocks)
        
        # 获取涨跌停状态
        limit_up, limit_down, suspended = self._get_limit_status(date_idx)
        # 确保形状匹配
        if limit_up.shape[0] != self.n_stocks:
            limit_up = np.zeros(self.n_stocks)
            limit_down = np.zeros(self.n_stocks)
            suspended = np.zeros(self.n_stocks)
        
        # 应用约束：
        # - 涨停股票不能买入（除非已有持仓且T+1已解锁）
        # - 跌停股票不能卖出
        # - 停牌股票不能交易
        # - 做空限制：权重不能为负
        
        # 先应用做空限制
        weights = np.maximum(weights, 0)
        
        # 应用可交易mask
        weights = weights * tradable
        
        # 对涨停股票特殊处理：如果权重增加（买入），则阻止
        # 这里简化处理：涨停股票权重设为0
        weights = weights * (1 - limit_up)
        
        # 归一化
        if np.sum(weights) > 1e-8:
            weights = weights / np.sum(weights)
        else:
            # 全部不可交易时，等权持有
            weights = tradable / np.sum(tradable) if np.sum(tradable) > 0 else \
                     np.ones(self.n_stocks) / self.n_stocks
        
        # 仓位上限约束
        weights = np.clip(weights, 0, self.max_pos)
        
        # 再次归一化
        if np.sum(weights) > 1e-8:
            weights = weights / np.sum(weights)
        
        return weights
    
    def _update_position_lock(self, old_weights: np.ndarray, new_weights: np.ndarray, date_idx: int):
        """
        更新T+1持仓锁定
        记录今日买入的股票，明日才能卖出
        """
        if date_idx >= len(self.dates) - 1:
            return
        
        next_date = self.dates[date_idx + 1]
        
        if self.position_lock is None:
            self.position_lock = {}
        
        # 找出今日加仓的股票
        for i in range(self.n_stocks):
            if new_weights[i] > old_weights[i] + 1e-6:  # 有加仓
                # 设置解锁日期为下一个交易日
                self.position_lock[i] = next_date
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        current_idx = self.current_step
        # 检查索引越界
        if current_idx >= len(self.dates):
            return self._get_state(len(self.dates) - 1) if len(self.dates) > 0 else np.zeros((self.n_stocks, self.lookback, self.n_features)), 0, True, {}
        
        # 🔥 关键检查：动作是否包含 NaN/Inf
        if np.isnan(action).any() or np.isinf(action).any():
            print(f"🔥 CRITICAL: Action contains NaN/Inf at step {current_idx}!")
            print(f"   Action stats: min={np.nanmin(action):.4f}, max={np.nanmax(action):.4f}")
            print(f"   NaN count: {np.isnan(action).sum()}, Inf count: {np.isinf(action).sum()}")
            # 用等权组合替代
            action = np.ones(self.n_stocks) / self.n_stocks
        
        current_date = self.dates[current_idx]
        
        # 动作->权重（确保非负且和为1）
        action = np.clip(action, a_min=1e-10, a_max=1.0)  # 防止负数或0
        action = action / action.sum()  # 重新归一化
        raw_weights = action
        
        # 应用约束
        weights = self._apply_constraints(raw_weights, current_idx + 1)
        
        # 更新T+1锁定
        self._update_position_lock(self.prev_weights, weights, current_idx)
        
        if current_idx + 1 >= len(self.dates):
            return self._get_state(current_idx), 0, True, {}
            
        next_date = self.dates[current_idx + 1]
        
        # 计算收益
        current_prices = self._price_cache.get(current_date, {})
        next_prices = self._price_cache.get(next_date, {})
        
        port_return = 0
        for stock, idx in self.stock_to_idx.items():
            if stock in current_prices and stock in next_prices:
                # 避免除零错误
                if current_prices[stock] <= 0:
                    continue
                ret = (next_prices[stock] - current_prices[stock]) / current_prices[stock]
                port_return += weights[idx] * ret
        
        turnover = np.sum(np.abs(weights - self.prev_weights))
        cost = turnover * self.tc
        net_return = port_return - cost
        
        # 奖励计算（风险调整后收益 - 核心改进）
        self.returns_buffer.append(net_return)
        if len(self.returns_buffer) > 60:
            self.returns_buffer.popleft()
        
        reward = 0
        sharpe = 0
        if len(self.returns_buffer) >= 30:
            mean_ret = np.mean(self.returns_buffer)
            std_ret = np.std(self.returns_buffer) + 1e-6
            sharpe = (mean_ret - self.cfg.trading.risk_free_rate) / std_ret * np.sqrt(252)
            
            # ========== 风险调整奖励方案选择 ==========
            # 通过 cfg.trading.reward_mode 选择奖励模式
            reward_mode = getattr(self.cfg.trading, 'reward_mode', 'sharpe_only')
            
            if reward_mode == 'sharpe_only':
                # 方案1: 纯夏普比率（最推荐）
                # 直接优化风险调整收益，避免过度冒险
                reward = sharpe
                
            elif reward_mode == 'sharpe_return_balanced':
                # 方案2: 夏普 + 收益平衡
                # 在保证风险可控的前提下追求收益
                reward = sharpe * 0.6 + net_return * 10  # 夏普占主导
                
            elif reward_mode == 'sortino':
                # 方案3: Sortino比率（只惩罚下行风险）
                downside_returns = [r for r in self.returns_buffer if r < 0]
                downside_std = np.std(downside_returns) + 1e-6 if downside_returns else 1e-6
                sortino = (mean_ret - self.cfg.trading.risk_free_rate) / downside_std * np.sqrt(252)
                reward = sortino
                
            elif reward_mode == 'calmar':
                # 方案4: Calmar比率（收益/最大回撤）
                if len(self.history) > 1:
                    peak = max(self.history)
                    current_val = self.history[-1]
                    max_dd = (peak - current_val) / peak if peak > current_val else 0
                    max_dd = max(max_dd, 0.01)  # 至少1%避免除0
                    annual_ret = mean_ret * 252
                    calmar = annual_ret / max_dd
                    reward = calmar * 0.1  # 缩放
                else:
                    reward = sharpe
                    
            elif reward_mode == 'risk_parity':
                # 方案5: 风险平价（惩罚波动率）
                reward = net_return * 10 - std_ret * 100  # 收益 - 风险惩罚
                
            else:  # default fallback
                reward = sharpe
            
            # 统一添加辅助惩罚项（所有模式通用）
            # 换手率惩罚（避免过度交易）
            turnover_penalty = turnover * getattr(self.cfg.trading, 'turnover_penalty', 0.01)
            reward -= turnover_penalty
            
            # 集中度惩罚（避免单股重仓）
            max_weight = np.max(weights)
            concentration_penalty = max(0, max_weight - 0.1) * getattr(self.cfg.trading, 'concentration_penalty', 0.1)
            reward -= concentration_penalty
            
        else:
            # Warmup阶段使用简单夏普近似
            reward = net_return * 10 if net_return > 0 else net_return * 20  # 亏损惩罚加倍
        
        # PPO 奖励缩放（根据reward_mode自动调整）
        reward_scale = getattr(self.cfg.trading, 'reward_scale', 1.0)
        reward = reward * reward_scale
        
        self.prev_weights = weights.copy()
        self.current_step += 1
        new_value = self.history[-1] * (1 + net_return)
        self.history.append(new_value)
        
        # 终止条件：数据结束或净值跌破50%
        done = (self.current_step >= len(self.dates) - 1) or (new_value < 0.5)
        
        # 获取当前市场状态
        limit_up, limit_down, suspended = self._get_limit_status(current_idx + 1)
        
        # 获取可交易掩码（用于PPO的Action Masking）
        action_mask = self._get_tradable_mask(current_idx + 1)
        
        # 检查是否有VLM数据（用于优先采样）
        vlm_dict = self._vlm_cache.get(next_date, {})
        has_vlm_data = any(vlm_dict.get(stock, False) for stock in self.stock_to_idx.keys())
        
        info = {
            'date': next_date,
            'portfolio_value': new_value,
            'daily_return': net_return,
            'turnover': turnover,
            'weights': weights.copy(),
            'sharpe': sharpe if len(self.returns_buffer) >= 30 else 0,
            'n_limit_up': int(limit_up.sum()),
            'n_limit_down': int(limit_down.sum()),
            'n_suspended': int(suspended.sum()),
            'has_vlm_data': has_vlm_data,
            'action_mask': action_mask  # (n_stocks,) 0/1 掩码，用于PPO Action Masking
        }
        
        return self._get_state(self.current_step), reward, done, info
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 确保 current_step 不超过数据长度，并至少保留1步用于评估
        max_valid_step = max(0, len(self.dates) - 2)  # 确保至少有1步可以step
        self.current_step = min(self.lookback, max_valid_step)
        if self.current_step < 0:
            self.current_step = 0
        self.prev_weights = np.ones(self.n_stocks) / self.n_stocks
        self.returns_buffer = deque(maxlen=60)
        self.history = [1.0]
        self.position_lock = {}  # 重置T+1锁定
        return self._get_state(self.current_step)
