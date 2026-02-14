import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import deque
import torch
from tqdm import tqdm
import os
from datetime import datetime


class RollingWalkForwardTrainer:
    """
    滚动微调步进回测训练器 (Rolling Walk-Forward with Fine-tuning)
    
    核心设计：
    1. 窗口划分：24月训练 + 1月验证 + 1月测试
    2. 增量学习：每轮继承上一轮权重，快速适应
    3. 早停机制：在验证集上选最佳模型
    4. 严格防泄露：每个窗口独立标准化
    
    示意图：
    Step 1: [2019.01-2020.12 Train][2021.01 Val][2021.02 Test]
    Step 2: [2019.02-2021.01 Train][2021.02 Val][2021.03 Test]
    Step 3: [2019.03-2021.02 Train][2021.03 Val][2021.04 Test]
    ...
    """
    
    def __init__(self, df: pd.DataFrame, train_config, data_config=None, 
                 trading_config=None, model_config=None, sac_config=None):
        self.df = df.copy()
        self.config = train_config
        self.data = data_config or train_config
        self.trading = trading_config
        self.model = model_config
        self.sac = sac_config
        
        # 日期处理
        self.df[self.data.date_col] = pd.to_datetime(self.df[self.data.date_col])
        self.df['year_month'] = self.df[self.data.date_col].dt.to_period('M')
        self.months = sorted(self.df['year_month'].unique())
        
        print(f"📊 Data loaded: {len(self.df)} records, {len(self.months)} months")
        print(f"📅 Date range: {self.df[self.data.date_col].min()} to {self.df[self.data.date_col].max()}")
        
        # 初始化日志和可视化
        from utils import TensorboardLogger, Visualizer, WindowedFeatureScaler
        self.logger = TensorboardLogger(train_config.log_dir)
        self.visualizer = Visualizer(train_config.plot_dir)
        self.scaler_class = WindowedFeatureScaler
        
        # 特征列
        self.feature_cols = [c for c in df.columns if c.startswith('f_')]
        
        # 状态追踪
        self.window_splits = []
        self.all_test_results = []
        self.global_step = 0
        
        # 增量学习状态
        self.prev_agent_state = None
        
    def get_window_split(self, window_idx: int) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]]:
        """
        获取当前窗口的数据切分
        
        Returns:
            (train_df, val_df, test_df, split_info)
        """
        train_months = self.data.train_window_months
        val_months = self.data.val_window_months
        test_months = self.data.test_window_months
        total_needed = train_months + val_months + test_months
        
        start_idx = window_idx * self.data.rolling_step_months
        end_idx = start_idx + total_needed
        
        if end_idx > len(self.months):
            return None
            
        # 切分月份
        train_months_list = self.months[start_idx:start_idx + train_months]
        val_months_list = self.months[start_idx + train_months:start_idx + train_months + val_months]
        test_months_list = self.months[start_idx + train_months + val_months:end_idx]
        
        # 切分数据
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
        """
        数据预处理：窗口内标准化（防止未来函数）
        Fit on Train, Transform on Val/Test
        """
        # 创建标准化器
        scaler = self.scaler_class(self.feature_cols)
        
        # 只在训练集上Fit
        train_scaled = scaler.fit_transform(train_df)
        
        # 对验证集和测试集Transform
        val_scaled = scaler.transform(val_df)
        test_scaled = scaler.transform(test_df)
        
        return train_scaled, val_scaled, test_scaled
    
    def create_agent(self, load_previous: bool = False):
        """创建智能体，可选择加载上一轮权重"""
        from environment import PortfolioTradingEnv
        from agent import PortfolioSAC
        from networks import create_networks
        
        # 创建配置包装器
        class ConfigWrapper:
            pass
        cfg_wrapper = ConfigWrapper()
        cfg_wrapper.data = self.data
        cfg_wrapper.trading = self.trading
        cfg_wrapper.model = self.model
        cfg_wrapper.sac = self.sac
        cfg_wrapper.train = self.config
        
        # 创建网络
        actor, critic1, critic2, tc1, tc2 = create_networks(
            self.data.n_stocks, 40, self.data.lookback_window,
            self.model.d_model, self.config.device
        )
        
        agent = PortfolioSAC(actor, critic1, critic2, tc1, tc2, cfg_wrapper, self.config.device)
        
        # 加载上一轮权重（增量学习）
        if load_previous and self.prev_agent_state is not None:
            agent.load_state_dict(self.prev_agent_state)
            print("🔄 Loaded weights from previous window (incremental learning)")
        
        return agent, cfg_wrapper
    
    def train_episode(self, env, agent, buffer, max_steps: int, 
                      current_window_start: int = 0) -> Tuple[float, List[Dict], int]:
        """训练一个episode"""
        state = env.reset()
        episode_reward = 0
        metrics = []
        step_count = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            # 检查是否有VLM数据（用于优先采样）
            has_vlm = info.get('has_vlm_data', False)
            
            # 存储经验
            buffer.store(state, action, reward, next_state, done, 
                        timestamp=step_count, has_vlm=has_vlm)
            
            # 更新网络
            loss_info = {}
            if len(buffer) >= self.sac.batch_size and step % self.sac.update_interval == 0:
                # 使用优先采样
                new_data_weight = self.config.new_data_sampling_weight if step_count > 0 else 1.0
                vlm_alpha = self.config.vlm_priority_alpha if self.config.use_priority_sampling else 0.0
                
                batch = buffer.sample(
                    self.sac.batch_size,
                    new_data_weight=new_data_weight,
                    vlm_priority_alpha=vlm_alpha,
                    current_window_start=current_window_start
                )
                
                if batch is not None:
                    loss_info = agent.update(batch)
                    if loss_info:
                        self.logger.log_training(
                            loss_info.get('critic_loss', 0),
                            loss_info.get('actor_loss', 0),
                            loss_info.get('entropy', 0),
                            self.global_step + step
                        )
            
            # 日志记录
            self.logger.log_step(
                info['portfolio_value'], info['daily_return'],
                info.get('sharpe', 0), info['turnover'],
                info['weights'], self.global_step + step
            )
            
            state = next_state
            episode_reward += reward
            metrics.append(info)
            step_count += 1
            
            if done:
                break
        
        self.global_step += step_count
        return episode_reward, metrics, step_count
    
    def evaluate(self, env, agent) -> Dict:
        """评估策略"""
        state = env.reset()
        done = False
        metrics = []
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            if info:
                metrics.append(info)
        
        if not metrics:
            return {
                'sharpe': 0,
                'total_return_pct': 0,
                'max_drawdown': 0,
                'turnover_mean': 0,
                'win_rate': 0,
                'metrics_history': []
            }
        
        returns = [m['daily_return'] for m in metrics]
        values = [m['portfolio_value'] for m in metrics]
        
        from utils import PerformanceMetrics
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        total_return = (values[-1] - 1) * 100
        max_dd = PerformanceMetrics.calculate_max_drawdown(values)
        
        turnover_mean = np.mean([m.get('turnover', 0) for m in metrics])
        win_rate = np.mean([r > 0 for r in returns])
        
        return {
            'sharpe': sharpe,
            'total_return_pct': total_return,
            'max_drawdown': max_dd,
            'turnover_mean': turnover_mean,
            'win_rate': win_rate,
            'metrics_history': metrics
        }
    
    def train_window(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, window_idx: int, 
                     is_first_window: bool) -> Tuple[Dict, Dict]:
        """
        训练一个窗口
        
        Returns:
            (val_result, test_result)
        """
        from environment import PortfolioTradingEnv
        from utils import EfficientReplayBuffer
        
        # 增量学习：非第一个窗口时加载上一轮权重
        load_previous = (not is_first_window) and self.config.inherit_weights
        agent, cfg_wrapper = self.create_agent(load_previous=load_previous)
        
        # 创建环境
        train_env = PortfolioTradingEnv(train_df, cfg_wrapper)
        val_env = PortfolioTradingEnv(val_df, cfg_wrapper)
        test_env = PortfolioTradingEnv(test_df, cfg_wrapper)
        
        # 创建回放缓冲区
        if is_first_window or not self.config.inherit_weights:
            buffer = EfficientReplayBuffer(
                self.sac.buffer_size, self.data.n_stocks,
                self.data.lookback_window, 40
            )
        else:
            # 保留部分旧buffer数据（防止灾难性遗忘）
            # 注意：这里简化处理，实际可能需要更复杂的buffer管理
            buffer = EfficientReplayBuffer(
                self.sac.buffer_size, self.data.n_stocks,
                self.data.lookback_window, 40
            )
        
        # Warmup
        print(f"\n🔥 Warmup phase ({self.config.warmup_steps} steps)...")
        state = train_env.reset()
        for i in range(min(self.config.warmup_steps, len(train_df))):
            action = np.random.randn(self.data.n_stocks)
            next_state, reward, done, _ = train_env.step(action)
            buffer.store(state, action, reward, next_state, done)
            state = next_state if not done else train_env.reset()
        
        # 确定训练轮数
        if is_first_window:
            n_episodes = self.config.episodes_first_window
            print(f"\n🚀 Cold start training ({n_episodes} episodes)...")
        else:
            n_episodes = self.config.episodes_finetune
            print(f"\n🔧 Fine-tuning ({n_episodes} episodes)...")
        
        # 训练循环
        best_val_sharpe = -np.inf
        best_model_state = None
        patience_counter = 0
        
        for episode in range(n_episodes):
            reward, metrics, steps = self.train_episode(
                train_env, agent, buffer, 
                self.config.max_steps_per_episode
            )
            
            # 计算训练集Sharpe
            returns = [m['daily_return'] for m in metrics]
            from utils import PerformanceMetrics
            train_sharpe = PerformanceMetrics.calculate_sharpe(returns)
            
            # 每N个episode在验证集上评估（早停）
            eval_interval = max(1, n_episodes // 20)
            if (episode + 1) % eval_interval == 0 or episode == n_episodes - 1:
                val_result = self.evaluate(val_env, agent)
                val_sharpe = val_result['sharpe']
                
                print(f"  Episode {episode+1}/{n_episodes}: "
                      f"Train Sharpe={train_sharpe:.2f}, Val Sharpe={val_sharpe:.2f}")
                
                # 早停检查：基于验证集表现
                if val_sharpe > best_val_sharpe + self.config.min_sharpe_improvement:
                    best_val_sharpe = val_sharpe
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = agent.get_state_dict()
                    print(f"    ✨ New best val sharpe: {best_val_sharpe:.2f}")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(f"    ⏹️  Early stopping at episode {episode+1}")
                        break
        
        # 加载验证集上表现最好的模型
        if best_model_state is not None:
            agent.load_state_dict(best_model_state)
            print(f"\n✅ Loaded best model (val sharpe: {best_val_sharpe:.2f})")
        
        # 保存当前窗口的模型状态（用于下一轮增量学习）
        self.prev_agent_state = agent.get_state_dict()
        
        # 最终验证集评估
        print(f"\n📊 Final Validation Evaluation:")
        val_result = self.evaluate(val_env, agent)
        print(f"  Sharpe:   {val_result['sharpe']:.2f}")
        print(f"  Return:   {val_result['total_return_pct']:.2f}%")
        print(f"  Max DD:   {val_result['max_drawdown']:.2%}")
        
        # 测试集评估（这才是记录的净值曲线）
        print(f"\n🎯 Test Evaluation (Paper Trading):")
        test_result = self.evaluate(test_env, agent)
        print(f"  Sharpe:   {test_result['sharpe']:.2f}")
        print(f"  Return:   {test_result['total_return_pct']:.2f}%")
        print(f"  Max DD:   {test_result['max_drawdown']:.2%}")
        print(f"  Turnover: {test_result['turnover_mean']:.2%}")
        print(f"  Win Rate: {test_result['win_rate']:.2%}")
        
        # 记录日志
        self.logger.log_evaluation(val_result['sharpe'], val_result['total_return_pct'], 
                                   val_result['max_drawdown'], window_idx, split='val')
        self.logger.log_evaluation(test_result['sharpe'], test_result['total_return_pct'], 
                                   test_result['max_drawdown'], window_idx, split='test')
        
        # 绘图
        self.visualizer.plot_training_report(val_result['metrics_history'], window_idx, split='val')
        self.visualizer.plot_training_report(test_result['metrics_history'], window_idx, split='test')
        
        # 保存模型
        if best_model_state is not None:
            save_path = f"{self.config.save_dir}/window{window_idx}_best.pt"
            torch.save(best_model_state, save_path)
        
        return val_result, test_result
    
    def run(self) -> List[Dict]:
        """
        执行滚动微调步进回测
        
        Returns:
            所有测试窗口的结果列表
        """
        print(f"\n{'='*70}")
        print(f"🎯 Rolling Walk-Forward Training with Fine-tuning")
        print(f"{'='*70}")
        print(f"Window: {self.data.train_window_months}m Train + "
              f"{self.data.val_window_months}m Val + "
              f"{self.data.test_window_months}m Test")
        print(f"Step: {self.data.rolling_step_months} month(s)")
        
        # 计算窗口数量
        total_months_per_window = (self.data.train_window_months + 
                                   self.data.val_window_months + 
                                   self.data.test_window_months)
        n_windows = (len(self.months) - total_months_per_window) // self.data.rolling_step_months + 1
        n_windows = max(0, n_windows)
        
        print(f"Total windows: {n_windows}")
        
        all_test_results = []
        window_splits = []
        
        for window_idx in range(n_windows):
            # 获取数据切分
            split_result = self.get_window_split(window_idx)
            if split_result is None:
                print(f"⚠️  Insufficient data for window {window_idx+1}, stopping")
                break
                
            train_df, val_df, test_df, split_info = split_result
            window_splits.append(split_info)
            
            # 记录窗口切分
            self.logger.log_window_split(
                window_idx,
                split_info['train_start'], split_info['train_end'],
                split_info['val_start'], split_info['val_end'],
                split_info['test_start'], split_info['test_end']
            )
            
            # 数据预处理（窗口内标准化，防止未来函数）
            train_scaled, val_scaled, test_scaled = self.prepare_data(train_df, val_df, test_df)
            
            # 检查数据量
            if len(train_scaled) < self.data.lookback_window * 2:
                print(f"⚠️  Insufficient training data ({len(train_scaled)} rows), skipping")
                continue
            
            # 训练当前窗口
            is_first_window = (window_idx == 0)
            val_result, test_result = self.train_window(
                train_scaled, val_scaled, test_scaled, 
                window_idx, is_first_window
            )
            
            all_test_results.append(test_result)
        
        # 绘制窗口时间线
        if window_splits:
            self.visualizer.plot_window_timeline(window_splits)
        
        # 绘制汇总
        if all_test_results:
            self.visualizer.plot_rolling_summary(all_test_results)
            
            # 打印汇总统计
            print(f"\n{'='*70}")
            print("📈 Rolling Window Summary (Test Results)")
            print(f"{'='*70}")
            
            sharpes = [r['sharpe'] for r in all_test_results]
            returns = [r['total_return_pct'] for r in all_test_results]
            drawdowns = [r['max_drawdown'] for r in all_test_results]
            turnovers = [r['turnover_mean'] for r in all_test_results]
            
            print(f"Sharpe:       {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}  "
                  f"[min: {np.min(sharpes):.2f}, max: {np.max(sharpes):.2f}]")
            print(f"Return (%):   {np.mean(returns):.2f} ± {np.std(returns):.2f}  "
                  f"[min: {np.min(returns):.2f}, max: {np.max(returns):.2f}]")
            print(f"Max DD:       {np.mean(drawdowns):.2%} ± {np.std(drawdowns):.2%}")
            print(f"Turnover:     {np.mean(turnovers):.2%} ± {np.std(turnovers):.2%}")
            
            # 累计净值
            cumulative_return = np.prod([(1 + r/100) for r in returns]) - 1
            print(f"Cumulative Return: {cumulative_return:.2%}")
        
        self.logger.close()
        return all_test_results
