import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Deque, Optional
from collections import deque
import torch
from torch.utils.tensorboard import SummaryWriter


class EfficientReplayBuffer:
    """
    内存高效型回放缓冲区（uint8量化，节省75%内存）
    支持：1) 保留部分旧buffer 2) 新数据高权重采样
    """
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
        
        # 时间戳和优先级（用于增量学习）
        self.timestamps = np.zeros(capacity, dtype=np.int32)
        self.has_vlm_data = np.zeros(capacity, dtype=np.bool_)
        self.priorities = np.ones(capacity, dtype=np.float32)
        
        self.running_mean = np.zeros((n_stocks, n_features))
        self.running_std = np.ones((n_stocks, n_features))
        self.count = 0
        
    def store(self, state: np.ndarray, action: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool, 
              timestamp: int = 0, has_vlm: bool = False):
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
        self.timestamps[self.ptr] = timestamp
        self.has_vlm_data[self.ptr] = has_vlm
        self.priorities[self.ptr] = 1.0  # 新数据默认优先级
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, 
               new_data_weight: float = 1.0,
               vlm_priority_alpha: float = 0.0,
               current_window_start: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        采样并反量化
        
        Args:
            batch_size: 批次大小
            new_data_weight: 新数据采样权重
            vlm_priority_alpha: VLM数据优先采样系数
            current_window_start: 当前窗口开始时间戳
        """
        if self.size == 0:
            return None
        
        # 确保 batch_size 不超过实际存储的样本数
        actual_batch_size = min(batch_size, self.size)
        if actual_batch_size < batch_size:
            print(f"Warning: Buffer size ({self.size}) < batch_size ({batch_size}), using {actual_batch_size}")
            
        # 计算采样概率
        probs = np.ones(self.size)
        
        # 新数据高权重采样
        if new_data_weight > 1.0 and current_window_start > 0:
            new_data_mask = self.timestamps[:self.size] >= current_window_start
            probs[new_data_mask] = new_data_weight
        
        # VLM数据优先采样
        if vlm_priority_alpha > 0:
            vlm_mask = self.has_vlm_data[:self.size]
            probs[vlm_mask] *= (1 + vlm_priority_alpha)
        
        # 归一化
        probs = probs / probs.sum()
        
        # 采样
        idxs = np.random.choice(self.size, actual_batch_size, replace=False, p=probs)
        
        # 反量化
        states = (self.states[idxs].astype(np.float32) / 255 - 0.5) * 10
        next_states = (self.next_states[idxs].astype(np.float32) / 255 - 0.5) * 10
        
        mean = self.running_mean[None, :, None, :]
        std = self.running_std[None, :, None, :] + 1e-8
        states = (states - mean) / std
        next_states = (next_states - mean) / std
        
        return (states, self.actions[idxs].astype(np.float32), self.rewards[idxs], 
                next_states, self.dones[idxs].astype(np.float32))
    
    def preserve_old_data(self, preserve_ratio: float = 0.3):
        """
        保留部分旧数据，防止灾难性遗忘
        随机保留preserve_ratio比例的数据
        """
        if self.size == 0:
            return
            
        n_preserve = int(self.size * preserve_ratio)
        if n_preserve == 0:
            self.clear()
            return
            
        # 随机选择要保留的索引
        preserve_idxs = np.random.choice(self.size, n_preserve, replace=False)
        
        # 临时存储
        temp_states = self.states[preserve_idxs].copy()
        temp_next_states = self.next_states[preserve_idxs].copy()
        temp_actions = self.actions[preserve_idxs].copy()
        temp_rewards = self.rewards[preserve_idxs].copy()
        temp_dones = self.dones[preserve_idxs].copy()
        temp_timestamps = self.timestamps[preserve_idxs].copy()
        temp_has_vlm = self.has_vlm_data[preserve_idxs].copy()
        
        # 清空并重新填充
        self.clear()
        for i in range(n_preserve):
            self.states[i] = temp_states[i]
            self.next_states[i] = temp_next_states[i]
            self.actions[i] = temp_actions[i]
            self.rewards[i] = temp_rewards[i]
            self.dones[i] = temp_dones[i]
            self.timestamps[i] = temp_timestamps[i]
            self.has_vlm_data[i] = temp_has_vlm[i]
        
        self.size = n_preserve
        self.ptr = n_preserve % self.capacity
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0
        self.count = 0
        self.running_mean = np.zeros((self.n_stocks, self.n_features))
        self.running_std = np.ones((self.n_stocks, self.n_features))
    
    def __len__(self):
        return self.size


class WindowedFeatureScaler:
    """
    窗口级别特征标准化器
    严格防止未来函数：每个窗口独立Fit，然后Transform到Val/Test
    """
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.mean = None
        self.std = None
        self.fitted = False
        
    def fit(self, df: pd.DataFrame):
        """
        在训练数据上计算统计量
        """
        available_cols = [c for c in self.feature_cols if c in df.columns]
        if not available_cols:
            print("Warning: No feature columns found for scaling")
            return self
            
        self.mean = df[available_cols].mean()
        self.std = df[available_cols].std() + 1e-8
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用标准化
        """
        if not self.fitted:
            print("Warning: Scaler not fitted, returning original data")
            return df
            
        df = df.copy()
        available_cols = [c for c in self.feature_cols if c in df.columns and c in self.mean.index]
        
        for col in available_cols:
            df[col] = (df[col] - self.mean[col]) / self.std[col]
            
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit then Transform"""
        return self.fit(df).transform(df)


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
    
    @staticmethod
    def calculate_calmar(returns: List[float], max_dd: float) -> float:
        """Calmar比率：年化收益/最大回撤"""
        if max_dd < 1e-8 or len(returns) < 30:
            return 0.0
        annual_return = np.mean(returns) * 252
        return annual_return / max_dd
    
    @staticmethod
    def calculate_sortino(returns: List[float], risk_free: float = 0.02/252) -> float:
        """Sortino比率：只考虑下行风险"""
        if len(returns) < 30:
            return 0.0
        returns = np.array(returns)
        excess = returns - risk_free
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) < 1e-8:
            return 0.0
        return np.mean(excess) / np.std(downside) * np.sqrt(252)


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
            
    def log_evaluation(self, eval_sharpe: float, eval_return: float, max_dd: float, window: int, 
                       split: str = "test"):
        """记录验证集或测试集表现"""
        self.writer.add_scalar(f'Eval/{split}/Sharpe', eval_sharpe, window)
        self.writer.add_scalar(f'Eval/{split}/Return', eval_return, window)
        self.writer.add_scalar(f'Eval/{split}/MaxDD', max_dd, window)
        
    def log_window_split(self, window: int, train_start: str, train_end: str,
                         val_start: str, val_end: str, test_start: str, test_end: str):
        """记录窗口切分信息"""
        self.writer.add_text(f'Window_{window}/Split', 
                            f"Train: {train_start} ~ {train_end}\n"
                            f"Val:   {val_start} ~ {val_end}\n"
                            f"Test:  {test_start} ~ {test_end}", window)
        
    def close(self):
        self.writer.close()


class Visualizer:
    """可视化"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def plot_training_report(self, metrics_history: List[Dict], window_id: int, split: str = "test"):
        """绘制训练报告"""
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
        plt.savefig(f'{self.save_dir}/{split}_win{window_id}.png', dpi=150)
        plt.close()
        
    def plot_rolling_summary(self, results: List[Dict]):
        """绘制滚动窗口汇总"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        sharpes = [r['sharpe'] for r in results]
        returns = [r['total_return_pct'] for r in results]
        drawdowns = [r['max_drawdown']*100 for r in results]
        turnovers = [r['turnover_mean']*100 for r in results]
        
        # Sharpe
        axes[0,0].bar(range(len(sharpes)), sharpes, color='steelblue')
        axes[0,0].axhline(y=np.mean(sharpes), color='red', linestyle='--', label=f'Mean: {np.mean(sharpes):.2f}')
        axes[0,0].set_title('Sharpe Ratio by Window')
        axes[0,0].set_xlabel('Window')
        axes[0,0].legend()
        
        # Return
        axes[0,1].bar(range(len(returns)), returns, color='green', alpha=0.7)
        axes[0,1].axhline(y=np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
        axes[0,1].set_title('Return % by Window')
        axes[0,1].set_xlabel('Window')
        axes[0,1].legend()
        
        # Max DD
        axes[1,0].bar(range(len(drawdowns)), drawdowns, color='red', alpha=0.7)
        axes[1,0].axhline(y=np.mean(drawdowns), color='blue', linestyle='--', label=f'Mean: {np.mean(drawdowns):.2f}%')
        axes[1,0].set_title('Max Drawdown % by Window')
        axes[1,0].set_xlabel('Window')
        axes[1,0].legend()
        
        # Turnover
        axes[1,1].bar(range(len(turnovers)), turnovers, color='orange', alpha=0.7)
        axes[1,1].axhline(y=np.mean(turnovers), color='red', linestyle='--', label=f'Mean: {np.mean(turnovers):.2f}%')
        axes[1,1].set_title('Turnover % by Window')
        axes[1,1].set_xlabel('Window')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/summary.png', dpi=150)
        plt.close()
        
    def plot_window_timeline(self, window_splits: List[Dict]):
        """可视化窗口时间线"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
        y_pos = 0
        
        for i, split in enumerate(window_splits):
            # 训练集
            train_start = pd.to_datetime(split['train_start'])
            train_end = pd.to_datetime(split['train_end'])
            ax.barh(y_pos, (train_end - train_start).days, left=train_start, 
                   color=colors['train'], alpha=0.7, label='Train' if i == 0 else "")
            
            # 验证集
            val_start = pd.to_datetime(split['val_start'])
            val_end = pd.to_datetime(split['val_end'])
            ax.barh(y_pos, (val_end - val_start).days, left=val_start, 
                   color=colors['val'], alpha=0.7, label='Validation' if i == 0 else "")
            
            # 测试集
            test_start = pd.to_datetime(split['test_start'])
            test_end = pd.to_datetime(split['test_end'])
            ax.barh(y_pos, (test_end - test_start).days, left=test_start, 
                   color=colors['test'], alpha=0.7, label='Test' if i == 0 else "")
            
            y_pos += 1
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Rolling Window')
        ax.set_title('Rolling Walk-Forward Window Timeline\n(24m Train + 1m Val → 1m Test)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/window_timeline.png', dpi=150)
        plt.close()
