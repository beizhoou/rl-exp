"""
PPO Agent V2 - 支持现金仓位和改进策略

核心改进:
1. 支持现金仓位 (n_stocks + 1)
2. 支持Softmax+Gaussian策略
3. 课程学习支持
4. 改进的数据清洗
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, NamedTuple
from collections import deque


class RolloutBufferSamples(NamedTuple):
    """Rollout Buffer 采样数据"""
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    action_masks: torch.Tensor


class RolloutBuffer:
    """
    V2: PPO Rollout Buffer - 支持现金仓位
    """
    def __init__(self, buffer_size: int, n_assets: int, 
                 lookback: int, n_features: int, device: str = 'cuda'):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.device = device
        
        # V2: n_assets = n_stocks + 1 (cash)
        self.observations = np.zeros((buffer_size, n_assets, lookback, n_features), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_assets), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, n_assets), dtype=np.float32)
        
    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        """添加一步数据"""
        if self.ptr >= self.buffer_size:
            return
        
        # 数据清洗
        if np.isnan(obs).any() or np.isinf(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(action).any() or np.isinf(action).any():
            n_assets = action.shape[0]
            action = np.ones(n_assets) / n_assets
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        if np.isnan(log_prob) or np.isinf(log_prob):
            log_prob = -10.0
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.action_masks[self.ptr] = action_mask
        
        self.ptr += 1
    
    def compute_returns_and_advantages(self, last_values: np.ndarray, 
                                       gamma: float = 0.99, 
                                       gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """使用GAE计算优势和回报"""
        actual_size = self.ptr
        advantages = np.zeros(actual_size, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(actual_size)):
            if t == actual_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + self.values[:actual_size]
        
        return advantages, returns
    
    def normalize_advantages(self, advantages: np.ndarray) -> np.ndarray:
        """优势归一化"""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def get(self, batch_size: int, advantages: np.ndarray, returns: np.ndarray):
        """生成mini-batch"""
        advantages = self.normalize_advantages(advantages)
        
        indices = np.random.permutation(self.ptr)
        
        start_idx = 0
        while start_idx < self.ptr:
            end_idx = min(start_idx + batch_size, self.ptr)
            batch_indices = indices[start_idx:end_idx]
            
            yield RolloutBufferSamples(
                observations=torch.FloatTensor(self.observations[batch_indices]).to(self.device),
                actions=torch.FloatTensor(self.actions[batch_indices]).to(self.device),
                old_log_probs=torch.FloatTensor(self.log_probs[batch_indices]).to(self.device),
                advantages=torch.FloatTensor(advantages[batch_indices]).to(self.device),
                returns=torch.FloatTensor(returns[batch_indices]).to(self.device),
                values=torch.FloatTensor(self.values[batch_indices]).to(self.device),
                action_masks=torch.FloatTensor(self.action_masks[batch_indices]).to(self.device),
            )
            
            start_idx = end_idx
    
    def clear(self):
        """清空buffer"""
        self.ptr = 0
    
    def __len__(self):
        return self.ptr


class PPOAgentV2:
    """
    V2: PPO Agent - 支持现金仓位和改进策略
    """
    def __init__(self, network, config, device='cuda'):
        self.network = network
        self.config = config
        self.device = device
        
        # V2: 检测策略类型
        self.policy_type = getattr(config.model, 'policy_distribution', 'dirichlet')
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.ppo.lr,
            eps=config.ppo.optimizer_eps,
            weight_decay=config.ppo.weight_decay
        )
        
        # 熵系数
        self.entropy_coef = config.ppo.entropy_coef
        self.entropy_coef_final = config.ppo.entropy_coef_final
        self.entropy_decay = config.ppo.entropy_coef_decay
        
        self.update_count = 0
        
        # V2: 课程学习状态
        self.curriculum_stage = 0
        
    def select_action(self, state: np.ndarray, action_mask: np.ndarray, 
                      deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        选择动作 - V2统一接口
        
        Returns:
            action: (n_assets,) 投资组合权重（含现金）
            log_prob: 动作对数概率
            value: 状态价值
        """
        with torch.no_grad():
            # 数据清洗
            if np.isnan(state).any() or np.isinf(state).any():
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            
            # V2: 使用统一接口
            action, log_prob, value, entropy = self.network.select_action(
                state_tensor, mask_tensor, deterministic
            )
            
            # 确保动作有效
            action_np = action.cpu().numpy()[0]
            action_np = np.clip(action_np, 1e-10, 1.0)
            action_np = action_np / action_np.sum()
            
            return (action_np, 
                    log_prob.cpu().item(), 
                    value.cpu().item())
    
    def update(self, buffer: RolloutBuffer, advantages: np.ndarray, 
               returns: np.ndarray) -> Dict[str, float]:
        """
        V2: PPO网络更新 - 支持两种策略类型
        """
        # 更新熵系数
        if self.entropy_decay:
            progress = min(1.0, self.update_count / 1000)
            self.entropy_coef = self.entropy_coef * (1 - progress) + self.entropy_coef_final * progress
        
        # 收集losses
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # 多轮更新
        for epoch in range(self.config.ppo.n_epochs):
            for batch in buffer.get(self.config.ppo.mini_batch_size, advantages, returns):
                # 数据清洗
                observations = torch.nan_to_num(batch.observations, nan=0.0, posinf=1.0, neginf=-1.0)
                actions = torch.clamp(batch.actions, min=1e-10, max=1.0)
                
                # V2: 评估动作
                result = self.network.evaluate_actions(observations, actions, batch.action_masks)
                
                if self.policy_type == 'dirichlet':
                    alpha, dist, new_log_probs, entropy, new_values = result
                else:  # softmax_gaussian
                    mu, dist, new_log_probs, entropy, new_values = result
                
                # 检查数值有效性
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    continue
                
                # 清理old_log_probs
                old_log_probs_clean = torch.nan_to_num(
                    batch.old_log_probs, nan=-10.0, posinf=0.0, neginf=-20.0
                )
                
                # ========== Policy Loss (PPO-Clip) ==========
                ratio = torch.exp(new_log_probs - old_log_probs_clean)
                ratio = torch.clamp(ratio, min=0.1, max=10.0)
                
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo.clip_range, 
                                   1 + self.config.ppo.clip_range) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                clip_fraction = (torch.abs(ratio - 1) > self.config.ppo.clip_range).float().mean()
                clip_fractions.append(clip_fraction.item())
                
                # ========== Value Loss ==========
                value_pred = new_values.squeeze()
                returns_batch = batch.returns
                
                # Returns归一化
                returns_mean = returns_batch.mean()
                returns_std = returns_batch.std() + 1e-8
                returns_normalized = (returns_batch - returns_mean) / returns_std
                
                value_pred_normalized = (value_pred - returns_mean) / returns_std
                value_old_normalized = (batch.values - returns_mean) / returns_std
                
                value_pred_clipped = value_old_normalized + torch.clamp(
                    value_pred_normalized - value_old_normalized,
                    -self.config.ppo.clip_range,
                    self.config.ppo.clip_range
                )
                value_loss1 = F.mse_loss(value_pred_normalized, returns_normalized)
                value_loss2 = F.mse_loss(value_pred_clipped, returns_normalized)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                
                # ========== Entropy Loss ==========
                entropy_loss = -entropy.mean()
                
                # ========== Total Loss ==========
                loss = (policy_loss + 
                       self.config.ppo.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.network.parameters(), 
                                        self.config.ppo.max_grad_norm)
                
                self.optimizer.step()
                
                # 记录losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # 清理显存
                del observations, actions, new_log_probs, entropy, new_values
                del ratio, surr1, surr2, policy_loss, value_loss, entropy_loss, loss
            
            torch.cuda.empty_cache()
        
        self.update_count += 1
        
        if len(policy_losses) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'entropy_coef': self.entropy_coef,
                'clip_fraction': 0.0,
                'approx_kl': 0.0,
            }
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'entropy_coef': self.entropy_coef,
            'clip_fraction': np.mean(clip_fractions),
            'approx_kl': 0.0,
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'entropy_coef': self.entropy_coef,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        self.entropy_coef = checkpoint.get('entropy_coef', self.config.ppo.entropy_coef)
    
    def load_state_dict(self, state_dict: dict):
        """从字典加载状态"""
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.entropy_coef = state_dict.get('entropy_coef', self.config.ppo.entropy_coef)
