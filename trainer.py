import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import deque
import torch
from tqdm import tqdm
import os

class RollingTrainer:
    """
    滚动窗口训练器
    
    训练流程：
    1. 每个窗口：12个月训练 -> 1个月测试
    2. 每个窗口训练5个episode
    3. 使用早停防止过拟合
    4. 保存每个窗口最优模型
    """
    def __init__(self, df: pd.DataFrame, train_config, data_config=None, 
                 trading_config=None, model_config=None, sac_config=None):
        self.df = df.copy()
        self.config = train_config
        self.data = data_config or train_config
        self.trading = trading_config
        self.model = model_config
        self.sac = sac_config
        
        self.df[self.data.date_col] = pd.to_datetime(self.df[self.data.date_col])
        self.df['year_month'] = self.df[self.data.date_col].dt.to_period('M')
        self.months = sorted(self.df['year_month'].unique())
        
        print(f"Data loaded: {len(self.df)} records, {len(self.months)} months")
        print(f"Date range: {self.df[self.data.date_col].min()} to {self.df[self.data.date_col].max()}")
        
        from utils import TensorboardLogger, Visualizer
        self.logger = TensorboardLogger(train_config.log_dir)
        self.visualizer = Visualizer(train_config.plot_dir)
        
    def get_split(self, window_idx: int):
        """获取当前窗口的训练集和测试集"""
        start_idx = window_idx
        end_idx = start_idx + self.data.train_months + self.data.test_months
        
        if end_idx > len(self.months):
            print(f"Window {window_idx+1}: Not enough data (need {end_idx} months, have {len(self.months)})")
            return None, None
            
        train_months = self.months[start_idx:start_idx + self.data.train_months]
        test_months = self.months[start_idx + self.data.train_months:end_idx]
        
        train_df = self.df[self.df['year_month'].isin(train_months)].copy()
        test_df = self.df[self.df['year_month'].isin(test_months)].copy()
        
        print(f"\nWindow {window_idx+1}:")
        print(f"  Train: {len(train_months)} months ({train_df[self.data.date_col].min()} to {train_df[self.data.date_col].max()})")
        print(f"  Test:  {len(test_months)} months ({test_df[self.data.date_col].min()} to {test_df[self.data.date_col].max()})")
        
        return train_df, test_df
    
    def train_episode(self, env, agent, buffer, max_steps: int, global_step: int, episode: int):
        """训练一个episode"""
        state = env.reset()
        episode_reward = 0
        metrics = []
        step_count = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            buffer.store(state, action, reward, next_state, done)
            
            # 更新网络
            loss_info = {}
            if len(buffer) >= self.sac.batch_size and step % self.sac.update_interval == 0:
                batch = buffer.sample(self.sac.batch_size)
                loss_info = agent.update(batch)
                if loss_info:
                    self.logger.log_training(
                        loss_info.get('critic_loss', 0),
                        loss_info.get('actor_loss', 0),
                        loss_info.get('entropy', 0),
                        global_step + step
                    )
            
            # 日志记录
            self.logger.log_step(
                info['portfolio_value'], info['daily_return'],
                info.get('sharpe', 0), info['turnover'],
                info['weights'], global_step + step
            )
            
            state = next_state
            episode_reward += reward
            metrics.append(info)
            step_count += 1
            
            if done:
                break
        
        return episode_reward, metrics, step_count
    
    def evaluate(self, env, agent):
        """评估策略"""
        state = env.reset()
        done = False
        metrics = []
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            metrics.append(info)
        
        # 计算评估指标
        returns = [m['daily_return'] for m in metrics]
        values = [m['portfolio_value'] for m in metrics]
        
        from utils import PerformanceMetrics
        sharpe = PerformanceMetrics.calculate_sharpe(returns)
        total_return = (values[-1] - 1) * 100
        max_dd = PerformanceMetrics.calculate_max_drawdown(values)
        
        # 计算额外指标
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
    
    def run(self, n_splits: Optional[int] = None):
        """
        执行滚动训练
        
        Args:
            n_splits: 滚动窗口数量，None则使用配置值
        """
        if n_splits is None:
            n_splits = self.config.n_splits
            
        all_results = []
        global_step = 0
        
        for window_idx in range(n_splits):
            print(f"\n{'='*70}")
            print(f"Rolling Window {window_idx+1}/{n_splits}")
            print(f"{'='*70}")
            
            train_df, test_df = self.get_split(window_idx)
            if train_df is None:
                break
            
            # 检查数据量
            if len(train_df) < self.data.lookback_window * 2:
                print(f"Skipping: insufficient training data ({len(train_df)} rows)")
                continue
            
            # 初始化环境、网络、智能体
            from environment import PortfolioTradingEnv
            from agent import PortfolioSAC
            from networks import create_networks
            from utils import EfficientReplayBuffer
            
            # 创建配置包装器传递给环境和智能体
            class ConfigWrapper:
                pass
            cfg_wrapper = ConfigWrapper()
            cfg_wrapper.data = self.data
            cfg_wrapper.trading = self.trading
            cfg_wrapper.model = self.model
            cfg_wrapper.sac = self.sac
            cfg_wrapper.train = self.config
            
            train_env = PortfolioTradingEnv(train_df, cfg_wrapper)
            
            actor, critic1, critic2, tc1, tc2 = create_networks(
                self.data.n_stocks, 40, self.data.lookback_window,
                self.model.d_model, self.config.device
            )
            
            agent = PortfolioSAC(actor, critic1, critic2, tc1, tc2, cfg_wrapper, self.config.device)
            
            buffer = EfficientReplayBuffer(
                self.sac.buffer_size, self.data.n_stocks,
                self.data.lookback_window, 40
            )
            
            # Warmup：随机探索填充回放缓冲区
            print(f"\nWarmup phase ({self.config.warmup_steps} steps)...")
            state = train_env.reset()
            for i in range(min(self.config.warmup_steps, len(train_df))):
                action = np.random.randn(self.data.n_stocks)
                next_state, reward, done, _ = train_env.step(action)
                buffer.store(state, action, reward, next_state, done)
                state = next_state if not done else train_env.reset()
            
            # 训练阶段
            print(f"\nTraining phase ({self.config.episodes_per_window} episodes)...")
            best_sharpe = -np.inf
            best_model_path = None
            patience_counter = 0
            
            for episode in range(self.config.episodes_per_window):
                reward, metrics, steps = self.train_episode(
                    train_env, agent, buffer, self.config.max_steps_per_episode,
                    global_step, episode
                )
                
                # 计算训练集表现
                returns = [m['daily_return'] for m in metrics]
                from utils import PerformanceMetrics
                train_sharpe = PerformanceMetrics.calculate_sharpe(returns)
                
                print(f"  Episode {episode+1}/{self.config.episodes_per_window}: "
                      f"Reward={reward:.4f}, Train Sharpe={train_sharpe:.2f}, Steps={steps}")
                
                global_step += steps
                
                # 早停检查
                if train_sharpe > best_sharpe + self.config.min_sharpe_improvement:
                    best_sharpe = train_sharpe
                    patience_counter = 0
                    # 保存最优模型
                    best_model_path = f"{self.config.save_dir}/window{window_idx}_best.pt"
                    agent.save(best_model_path)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        print(f"  Early stopping at episode {episode+1}")
                        break
            
            # 加载最优模型进行评估
            if best_model_path and os.path.exists(best_model_path):
                agent.load(best_model_path)
                print(f"\nLoaded best model for evaluation")
            
            # 评估阶段
            print(f"\nEvaluation phase...")
            test_env = PortfolioTradingEnv(test_df, cfg_wrapper)
            result = self.evaluate(test_env, agent)
            
            # 打印结果
            print(f"\nTest Results:")
            print(f"  Sharpe:      {result['sharpe']:.2f}")
            print(f"  Return:      {result['total_return_pct']:.2f}%")
            print(f"  Max DD:      {result['max_drawdown']:.2%}")
            print(f"  Turnover:    {result['turnover_mean']:.2%}")
            print(f"  Win Rate:    {result['win_rate']:.2%}")
            
            # 记录日志
            self.logger.log_evaluation(result['sharpe'], result['total_return_pct'], 
                                      result['max_drawdown'], window_idx)
            self.visualizer.plot_training_report(result['metrics_history'], window_idx)
            
            all_results.append(result)
            
            # 保存最终模型
            final_path = f"{self.config.save_dir}/window{window_idx}_final.pt"
            agent.save(final_path)
        
        # 汇总所有窗口结果
        self.visualizer.plot_rolling_summary(all_results)
        
        # 打印汇总统计
        print(f"\n{'='*70}")
        print("Rolling Window Summary")
        print(f"{'='*70}")
        
        sharpes = [r['sharpe'] for r in all_results]
        returns = [r['total_return_pct'] for r in all_results]
        drawdowns = [r['max_drawdown'] for r in all_results]
        turnovers = [r['turnover_mean'] for r in all_results]
        
        print(f"Sharpe:       {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}  "
              f"[min: {np.min(sharpes):.2f}, max: {np.max(sharpes):.2f}]")
        print(f"Return (%):   {np.mean(returns):.2f} ± {np.std(returns):.2f}  "
              f"[min: {np.min(returns):.2f}, max: {np.max(returns):.2f}]")
        print(f"Max DD:       {np.mean(drawdowns):.2%} ± {np.std(drawdowns):.2%}")
        print(f"Turnover:     {np.mean(turnovers):.2%} ± {np.std(turnovers):.2%}")
        
        self.logger.close()
        return all_results
