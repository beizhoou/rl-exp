"""
PPO Trainer V2 - 滚动训练 + 课程学习

核心改进:
1. 支持现金仓位
2. 课程学习：分阶段增加难度
3. 调试模式：0交易成本快速验证
4. 改进的评估指标
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import os

from ppo_agent_v2 import PPOAgentV2, RolloutBuffer
from networks_v2 import create_networks_v2
from utils import TensorboardLogger, Visualizer, WindowedFeatureScaler, PerformanceMetrics


class PPOTrainerV2:
    """
    V2: PPO训练器 - 支持现金仓位和课程学习
    """
    
    def __init__(self, df: pd.DataFrame, train_config, data_config=None, 
                 trading_config=None, model_config=None, ppo_config=None):
        self.df = df.copy()
        self.config = train_config
        self.data = data_config or train_config
        self.trading = trading_config
        self.model = model_config
        self.ppo = ppo_config
        
        # 日期处理
        self.df[self.data.date_col] = pd.to_datetime(self.df[self.data.date_col])
        self.df['year_month'] = self.df[self.data.date_col].dt.to_period('M')
        self.months = sorted(self.df['year_month'].unique())
        
        print(f"📊 Data loaded: {len(self.df)} records, {len(self.months)} months")
        print(f"📅 Date range: {self.df[self.data.date_col].min()} to {self.df[self.data.date_col].max()}")
        
        # V2: 现金仓位
        self.enable_cash = getattr(self.trading, 'enable_cash', True)
        self.n_assets = self.data.n_stocks + (1 if self.enable_cash else 0)
        print(f"💰 Cash position: {'Enabled' if self.enable_cash else 'Disabled'}")
        print(f"   Total assets: {self.n_assets} ({self.data.n_stocks} stocks + {1 if self.enable_cash else 0} cash)")
        
        # 初始化日志和可视化
        self.logger = TensorboardLogger(train_config.log_dir)
        self.visualizer = Visualizer(train_config.plot_dir)
        
        # 特征列
        self.feature_cols = [c for c in df.columns if c.startswith('f_')]
        self.n_features = len(self.feature_cols)
        print(f"Feature dimensions: {self.n_features}")
        
        # 状态追踪
        self.window_splits = []
        self.all_test_results = []
        self.global_step = 0
        
        # 增量学习状态
        self.prev_agent_state = None
        
        # V2: 课程学习状态
        self.curriculum_stage = 0
        self._apply_curriculum_stage(0)
    
    def _apply_curriculum_stage(self, stage: int):
        """V2: 应用课程学习阶段"""
        stages = getattr(self.trading, 'curriculum_stages', None)
        if stages is None or stage >= len(stages):
            return
        
        cfg = stages[stage]
        self.trading.reward_mode = cfg.get('reward_mode', 'risk_adjusted')
        self.trading.transaction_cost = cfg.get('cost', 0.0015)
        
        print(f"\n🎓 Curriculum Stage {stage + 1}/{len(stages)}")
        print(f"   Reward Mode: {self.trading.reward_mode}")
        print(f"   Transaction Cost: {self.trading.transaction_cost:.4%}")
    
    def _check_curriculum_progress(self, update: int, val_sharpe: float):
        """V2: 检查是否进入下一阶段"""
        stages = getattr(self.trading, 'curriculum_stages', None)
        if stages is None:
            return
        
        if self.curriculum_stage >= len(stages) - 1:
            return
        
        current_cfg = stages[self.curriculum_stage]
        min_updates = current_cfg.get('min_updates', 50)
        
        if update >= min_updates and val_sharpe > 0.5:  # Sharpe > 0.5视为学会
            self.curriculum_stage += 1
            self._apply_curriculum_stage(self.curriculum_stage)
    
    def get_window_split(self, window_idx: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]]:
        """获取当前窗口的数据切分"""
        train_months = self.data.train_window_months
        val_months = self.data.val_window_months
        test_months = self.data.test_window_months
        total_needed = train_months + val_months + test_months
        
        start_idx = window_idx * self.data.rolling_step_months
        end_idx = start_idx + total_needed
        
        if end_idx > len(self.months):
            return None
        
        train_months_list = self.months[start_idx:start_idx + train_months]
        val_months_list = self.months[start_idx + train_months:start_idx + train_months + val_months]
        test_months_list = self.months[start_idx + train_months + val_months:end_idx]
        
        train_df = self.df[self.df['year_month'].isin(train_months_list)].copy()
        val_df = self.df[self.df['year_month'].isin(val_months_list)].copy()
        test_df = self.df[self.df['year_month'].isin(test_months_list)].copy()
        
        split_info = {
            'window_idx': window_idx,
            'train_start': train_df[self.data.date_col].min(),
            'train_end': train_df[self.data.date_col].max(),
            'val_start': val_df[self.data.date_col].min() if len(val_df) > 0 else None,
            'val_end': val_df[self.data.date_col].max() if len(val_df) > 0 else None,
            'test_start': test_df[self.data.date_col].min() if len(test_df) > 0 else None,
            'test_end': test_df[self.data.date_col].max() if len(test_df) > 0 else None,
        }
        
        print(f"\n{'='*70}")
        print(f"📦 Window {window_idx+1}")
        print(f"{'='*70}")
        print(f"  Train: {len(train_months_list)} months ({split_info['train_start']} to {split_info['train_end']})")
        print(f"  Val:   {len(val_months_list)} month  ({split_info['val_start']} to {split_info['val_end']})")
        print(f"  Test:  {len(test_months_list)} month  ({split_info['test_start']} to {split_info['test_end']})")
        
        return train_df, val_df, test_df, split_info
    
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """数据预处理"""
        scaler = WindowedFeatureScaler(self.feature_cols)
        train_scaled = scaler.fit_transform(train_df)
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)
        return train_scaled, val_scaled, test_scaled
    
    def collect_rollout(self, env, agent, buffer, n_steps: int):
        """收集PPO Rollout数据"""
        from environment_v2 import PortfolioTradingEnv
        
        state = env.reset()
        episode_info = {
            'rewards': [],
            'values': [],
            'portfolio_values': [],
            'cash_weights': [],  # V2: 追踪现金仓位
        }
        
        for step in range(n_steps):
            action_mask = env._get_tradable_mask(env.current_step)
            
            action, log_prob, value = agent.select_action(state, action_mask, deterministic=False)
            
            next_state, reward, done, info = env.step(action)
            
            buffer.add(state, action, reward, value, log_prob, done, action_mask)
            
            episode_info['rewards'].append(reward)
            episode_info['values'].append(value)
            episode_info['portfolio_values'].append(info['portfolio_value'])
            episode_info['cash_weights'].append(info.get('cash_weight', 0))  # V2
            
            self.logger.log_step(
                info['portfolio_value'], info['daily_return'],
                0, info['turnover'],
                info['weights'], self.global_step + step
            )
            
            state = next_state if not done else env.reset()
            
            if done:
                break
        
        self.global_step += len(episode_info['rewards'])
        return episode_info
    
    def evaluate(self, env, agent):
        """V2: 评估策略（包含现金仓位指标）"""
        from environment_v2 import PortfolioTradingEnv
        
        state = env.reset()
        done = False
        metrics = []
        
        while not done:
            action_mask = env._get_tradable_mask(env.current_step)
            action, _, _ = agent.select_action(state, action_mask, deterministic=True)
            state, reward, done, info = env.step(action)
            if info:
                metrics.append(info)
        
        if not metrics:
            return {'sharpe': 0, 'total_return_pct': 0, 'max_drawdown': 0, 
                   'turnover_mean': 0, 'win_rate': 0, 'avg_cash_weight': 0, 'metrics_history': []}
        
        returns = [m['daily_return'] for m in metrics]
        values = [m['portfolio_value'] for m in metrics]
        cash_weights = [m.get('cash_weight', 0) for m in metrics]
        
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        total_return = (values[-1] - 1) * 100
        max_dd = PerformanceMetrics.calculate_max_drawdown(values)
        turnover_mean = np.mean([m.get('turnover', 0) for m in metrics])
        win_rate = np.mean([r > 0 for r in returns])
        avg_cash = np.mean(cash_weights)
        
        return {
            'sharpe': sharpe,
            'total_return_pct': total_return,
            'max_drawdown': max_dd,
            'turnover_mean': turnover_mean,
            'win_rate': win_rate,
            'avg_cash_weight': avg_cash,  # V2: 平均现金仓位
            'metrics_history': metrics
        }
    
    def train_window(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, window_idx: int, 
                     is_first_window: bool):
        """V2: 训练一个窗口 - 支持课程学习"""
        from environment_v2 import PortfolioTradingEnv
        
        # 创建配置包装器
        class ConfigWrapper:
            pass
        cfg_wrapper = ConfigWrapper()
        cfg_wrapper.data = self.data
        cfg_wrapper.trading = self.trading
        cfg_wrapper.model = self.model
        cfg_wrapper.ppo = self.ppo
        cfg_wrapper.train = self.config
        
        # V2: 创建环境（支持现金仓位）
        train_env = PortfolioTradingEnv(train_df, cfg_wrapper, enable_cash=self.enable_cash)
        val_env = PortfolioTradingEnv(val_df, cfg_wrapper, enable_cash=self.enable_cash)
        test_env = PortfolioTradingEnv(test_df, cfg_wrapper, enable_cash=self.enable_cash)
        
        # V2: 创建网络（支持现金仓位和新策略）
        network = create_networks_v2(
            n_stocks=self.data.n_stocks,
            n_features=self.n_features,
            lookback_window=self.data.lookback_window,
            d_model=self.model.d_model,
            temperature=self.model.temperature,
            device=self.config.device,
            enable_cash=self.enable_cash,
            policy_type=getattr(self.model, 'policy_distribution', 'softmax_gaussian'),
            top_k=getattr(self.model, 'top_k', None)
        )
        
        # V2: 创建Agent
        agent = PPOAgentV2(network, cfg_wrapper, self.config.device)
        
        # 加载上一轮权重
        if not is_first_window and self.config.inherit_weights and self.prev_agent_state:
            agent.load_state_dict(self.prev_agent_state)
            print("🔄 加载上一轮权重（增量学习）")
        
        # V2: 创建Rollout Buffer（n_assets代替n_stocks）
        buffer = RolloutBuffer(
            self.ppo.batch_size, self.n_assets,
            self.data.lookback_window, self.n_features, self.config.device
        )
        
        # 训练循环
        n_updates = self.config.total_timesteps_per_window // self.ppo.batch_size
        best_val_sharpe = -np.inf
        best_model_state = None
        patience_counter = 0
        
        print(f"\n🚀 开始训练 (每窗口 {n_updates} 次更新)")
        print(f"   课程阶段: {self.curriculum_stage + 1}")
        print(f"   奖励模式: {self.trading.reward_mode}")
        print(f"   交易成本: {self.trading.transaction_cost:.4%}")
        
        for update in range(n_updates):
            # 1. 收集Rollout
            episode_info = self.collect_rollout(
                train_env, agent, buffer, self.ppo.batch_size
            )
            
            # 2. 计算GAE
            last_state = train_env._get_state(train_env.current_step)
            last_mask = train_env._get_tradable_mask(train_env.current_step)
            with torch.no_grad():
                last_state_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.config.device)
                last_mask_t = torch.FloatTensor(last_mask).unsqueeze(0).to(self.config.device)
                _, _, last_value, _ = network.select_action(last_state_t, last_mask_t)
                last_value = last_value.cpu().numpy()[0, 0]
            
            advantages, returns = buffer.compute_returns_and_advantages(
                last_value, self.ppo.gamma, self.ppo.gae_lambda
            )
            
            # 3. PPO更新
            loss_info = agent.update(buffer, advantages, returns)
            
            # 清空buffer
            buffer.clear()
            torch.cuda.empty_cache()
            
            # 记录训练指标
            avg_reward = np.mean(episode_info['rewards'])
            avg_cash = np.mean(episode_info['cash_weights'])
            self.logger.log_training(
                loss_info['value_loss'],
                loss_info['policy_loss'],
                loss_info['entropy_loss'],
                self.global_step,
                reward=avg_reward
            )
            
            # 打印进度
            if (update + 1) % max(1, n_updates // 10) == 0:
                print(f"  Update {update+1}/{n_updates}: "
                      f"Reward={avg_reward:.4f}, "
                      f"Cash={avg_cash:.2%}, "
                      f"Policy Loss={loss_info['policy_loss']:.4f}, "
                      f"Value Loss={loss_info['value_loss']:.4f}")
            
            # 验证和早停
            if (update + 1) % max(1, n_updates // 5) == 0:
                val_result = self.evaluate(val_env, agent)
                val_sharpe = val_result['sharpe']
                
                print(f"    📊 Val Sharpe: {val_sharpe:.2f}, Cash: {val_result['avg_cash_weight']:.2%}")
                
                # V2: 课程学习进度检查
                self._check_curriculum_progress(update, val_sharpe)
                
                if val_sharpe > best_val_sharpe + self.config.min_sharpe_improvement:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0
                    best_model_state = {
                        'network': agent.network.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'entropy_coef': agent.entropy_coef,
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(f"    ⏹️  早停于 Update {update+1}")
                        break
        
        # 加载验证集最优模型
        if best_model_state:
            agent.network.load_state_dict(best_model_state['network'])
            print(f"\n✅ 加载最优模型 (Val Sharpe: {best_val_sharpe:.2f})")
        
        # 保存当前窗口权重
        self.prev_agent_state = {
            'network': agent.network.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'entropy_coef': agent.entropy_coef,
        }
        
        # 最终评估
        print(f"\n📊 验证集表现:")
        val_result = self.evaluate(val_env, agent)
        print(f"  Sharpe: {val_result['sharpe']:.2f}")
        print(f"  Avg Cash: {val_result['avg_cash_weight']:.2%}")
        
        print(f"\n🎯 测试集表现:")
        test_result = self.evaluate(test_env, agent)
        print(f"  Sharpe: {test_result['sharpe']:.2f}")
        print(f"  Return: {test_result['total_return_pct']:.2f}%")
        print(f"  Max DD: {test_result['max_drawdown']:.2%}")
        print(f"  Avg Cash: {test_result['avg_cash_weight']:.2%}")
        
        # 记录日志
        self.logger.log_evaluation(val_result['sharpe'], val_result['total_return_pct'], 
                                   val_result['max_drawdown'], window_idx, split='val')
        self.logger.log_evaluation(test_result['sharpe'], test_result['total_return_pct'], 
                                   test_result['max_drawdown'], window_idx, split='test')
        
        return val_result, test_result
    
    def run(self):
        """V2: 执行滚动训练"""
        print(f"\n{'='*70}")
        print(f"🎯 PPO V2 Rolling Walk-Forward Training")
        print(f"{'='*70}")
        print(f"Algorithm: PPO-Clip with Cash Position")
        print(f"Cash Enabled: {self.enable_cash}")
        print(f"Reward Mode: {self.trading.reward_mode}")
        print(f"Debug Zero Cost: {getattr(self.trading, 'debug_zero_cost', False)}")
        print(f"Policy Type: {getattr(self.model, 'policy_distribution', 'softmax_gaussian')}")
        print(f"GAE Lambda: {self.ppo.gae_lambda}")
        print(f"Clip Range: {self.ppo.clip_range}")
        
        # 计算窗口数量
        total_months = (self.data.train_window_months + 
                       self.data.val_window_months + 
                       self.data.test_window_months)
        n_windows = max(0, (len(self.months) - total_months) // self.data.rolling_step_months + 1)
        
        # 应用最大窗口数限制
        if hasattr(self.config, 'max_windows') and self.config.max_windows:
            n_windows = min(n_windows, self.config.max_windows)
            print(f"⚡ Quick mode: limited to {n_windows} windows")
        
        print(f"Total windows: {n_windows}")
        
        all_test_results = []
        
        for window_idx in range(n_windows):
            split_result = self.get_window_split(window_idx)
            if split_result is None:
                break
            
            train_df, val_df, test_df, split_info = split_result
            
            # 数据预处理
            train_scaled, val_scaled, test_scaled = self.prepare_data(train_df, val_df, test_df)
            
            if len(train_scaled) < self.data.lookback_window * 2:
                print(f"⚠️  训练数据不足，跳过")
                continue
            
            # 训练当前窗口
            is_first_window = (window_idx == 0)
            val_result, test_result = self.train_window(
                train_scaled, val_scaled, test_scaled, 
                window_idx, is_first_window
            )
            
            all_test_results.append(test_result)
        
        # 汇总结果
        if all_test_results:
            self.visualizer.plot_rolling_summary(all_test_results)
            
            print(f"\n{'='*70}")
            print("📈 Rolling Window Summary (Test Results)")
            print(f"{'='*70}")
            
            sharpes = [r['sharpe'] for r in all_test_results]
            returns = [r['total_return_pct'] for r in all_test_results]
            drawdowns = [r['max_drawdown'] for r in all_test_results]
            cash_weights = [r.get('avg_cash_weight', 0) for r in all_test_results]
            
            print(f"Sharpe:       {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
            print(f"Return (%):   {np.mean(returns):.2f} ± {np.std(returns):.2f}")
            print(f"Max DD:       {np.mean(drawdowns):.2%} ± {np.std(drawdowns):.2%}")
            print(f"Avg Cash:     {np.mean(cash_weights):.2%} ± {np.std(cash_weights):.2%}")
        
        self.logger.close()
        return all_test_results
