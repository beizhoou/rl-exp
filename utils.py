import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Deque
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter

class EfficientReplayBuffer:
    """内存高效型回放缓冲区（uint8量化，节省75%内存）"""
    def __init__(self, capacity: int, n_stocks: int, lookback: int, n_features: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.n_stocks = n_stocks
        self.lookback = lookback
        self.n_features = n_features
        
        # uint8存储
        self.states = np.zeros((capacity, n_stocks, lookback, n_features), dtype=np.uint8)
        self.next_states = np.zeros((capacity, n_stocks, lookback, n_features), dtype=np.uint8)
        self.actions = np.zeros((capacity, n_stocks), dtype=np.float16)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
        self.running_mean = np.zeros((n_stocks, n_features))
        self.running_std = np.ones((n_stocks, n_features))
        self.count = 0
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool):
        """存储时量化"""
        self.count += 1
        delta = state.mean(axis=1) - self.running_mean
        self.running_mean += delta / self.count
        delta2 = state.mean(axis=1) - self.running_mean
        self.running_std = np.sqrt((self.running_std**2 * (self.count-1) + delta * delta2) / self.count)
        
        state_uint8 = np.clip((state / 10 + 0.5) * 255, 0, 255).astype(np.uint8)
        next_state_uint8 = np.clip((next_state / 10 + 0.5) * 255, 0, 255).astype(np.uint8)
        
        self.states[self.ptr] = state_uint8
        self.actions[self.ptr] = action.astype(np.float16)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state_uint8
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """采样并反量化"""
        idxs = np.random.choice(self.size, batch_size, replace=False)
        
        states = (self.states[idxs].astype(np.float32) / 255 - 0.5) * 10
        next_states = (self.next_states[idxs].astype(np.float32) / 255 - 0.5) * 10
        
        mean = self.running_mean[None, :, None, :]
        std = self.running_std[None, :, None, :] + 1e-8
        states = (states - mean) / std
        next_states = (next_states - mean) / std
        
        return (states, self.actions[idxs].astype(np.float32), self.rewards[idxs], 
                next_states, self.dones[idxs].astype(np.float32))
    
    def __len__(self):
        return self.size

class PerformanceMetrics:
    """性能指标计算"""
    @staticmethod
    def calculate_sharpe(returns: List[float], risk_free: float = 0.02/252) -> float:
        if len(returns) < 30:
            return 0.0
        returns = np.array(returns)
        excess = returns - risk_free
        if np.std(excess) < 1e-8:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: List[float]) -> float:
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)

class TensorboardLogger:
    """训练日志"""
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_training(self, critic_loss: float, actor_loss: float, entropy: float, step: int):
        self.writer.add_scalar('Loss/Critic', critic_loss, step)
        self.writer.add_scalar('Loss/Actor', actor_loss, step)
        self.writer.add_scalar('Train/Entropy', entropy, step)
        
    def log_step(self, portfolio_value: float, daily_return: float, 
                 sharpe: float, turnover: float, weights: np.ndarray, step: int):
        self.writer.add_scalar('Portfolio/Value', portfolio_value, step)
        self.writer.add_scalar('Portfolio/Daily_Return', daily_return, step)
        self.writer.add_scalar('Portfolio/Sharpe', sharpe, step)
        self.writer.add_scalar('Portfolio/Turnover', turnover, step)
        
        hhi = np.sum(weights ** 2)
        self.writer.add_scalar('Portfolio/HHI', hhi, step)
        
        if step % 100 == 0:
            self.writer.add_histogram('Weights/Dist', weights, step)
            
    def log_evaluation(self, eval_sharpe: float, eval_return: float, max_dd: float, window: int):
        self.writer.add_scalar('Eval/Sharpe', eval_sharpe, window)
        self.writer.add_scalar('Eval/Return', eval_return, window)
        self.writer.add_scalar('Eval/MaxDD', max_dd, window)
        
    def close(self):
        self.writer.close()

class Visualizer:
    """可视化"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def plot_training_report(self, metrics_history: List[Dict], window_id: int):
        # 如果没有有效数据，跳过绘图
        if not metrics_history:
            print(f"Warning: No valid metrics for window {window_id}, skipping plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        dates = range(len(metrics_history))
        values = [m['portfolio_value'] for m in metrics_history]

        axes[0,0].plot(dates, values, label='Portfolio')
        axes[0,0].set_title('Cumulative Return')
        axes[0,0].grid(True, alpha=0.3)

        returns = [m['daily_return'] for m in metrics_history]
        rolling_sharpe = pd.Series(returns).rolling(30).mean() / \
                        (pd.Series(returns).rolling(30).std() + 1e-8) * np.sqrt(252)
        axes[0,1].plot(dates, rolling_sharpe, color='green')
        axes[0,1].set_title('Rolling Sharpe')
        axes[0,1].grid(True, alpha=0.3)

        cummax = np.maximum.accumulate(values)
        drawdown = (cummax - values) / cummax
        axes[1,0].fill_between(dates, -drawdown*100, 0, color='red', alpha=0.3)
        max_dd = max(drawdown) if len(drawdown) > 0 else 0
        axes[1,0].set_title(f'Max DD: {max_dd:.2%}')

        turnovers = [m.get('turnover', 0) for m in metrics_history]
        if turnovers:
            axes[1,1].hist(turnovers, bins=30, alpha=0.7)
        axes[1,1].set_title(f'Turnover Mean: {np.mean(turnovers):.2%}')

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/train_win{window_id}.png', dpi=150)
        plt.close()
        
    def plot_rolling_summary(self, results: List[Dict]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sharpes = [r['sharpe'] for r in results]
        returns = [r['total_return_pct'] for r in results]
        drawdowns = [r['max_drawdown']*100 for r in results]
        
        axes[0].bar(range(len(sharpes)), sharpes, color='steelblue')
        axes[0].axhline(y=np.mean(sharpes), color='red', linestyle='--')
        axes[0].set_title(f'Sharpe (Avg: {np.mean(sharpes):.2f})')
        
        axes[1].bar(range(len(returns)), returns, color='green')
        axes[1].set_title(f'Return % (Avg: {np.mean(returns):.2f})')
        
        axes[2].bar(range(len(drawdowns)), drawdowns, color='red')
        axes[2].set_title(f'Max DD % (Avg: {np.mean(drawdowns):.2f})')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/summary.png', dpi=150)
        plt.close()