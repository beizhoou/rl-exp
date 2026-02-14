"""
PPO (Proximal Policy Optimization) Agent
========================================

针对 A 股投资组合管理任务的 PPO 实现

核心特性：
1. On-policy 策略：Collect -> GAE -> Update -> Clear
2. Dirichlet 分布：天然满足权重约束（sum=1, w>=0）
3. Action Masking：停牌/跌停股票浓度参数设为极小值
4. Advantage Normalization：稳定训练
5. 输入数据清洗：防止 NaN/Inf 传播
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
    PPO Rollout Buffer
    
    存储 trajectory 数据，用于 on-policy 更新
    """
    def __init__(self, buffer_size: int, n_stocks: int, 
                 lookback: int, n_features: int, device: str = 'cuda'):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.device = device
        
        # 存储空间
        self.observations = np.zeros((buffer_size, n_stocks, lookback, n_features), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_stocks), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, n_stocks), dtype=np.float32)
        
    def add(self, obs, action, reward, value, log_prob, done, action_mask):
        """添加一步数据"""
        if self.ptr >= self.buffer_size:
            return
        
        # 数据清洗：检查 NaN/Inf
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"⚠️ Warning: Observation contains NaN/Inf at step {self.ptr}")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if np.isnan(action).any() or np.isinf(action).any():
            print(f"⚠️ Warning: Action contains NaN/Inf at step {self.ptr}")
            action = np.nan_to_num(action, nan=1.0/n_stocks, posinf=1.0, neginf=0.0)
            # 重新归一化
            action = action / (action.sum() + 1e-8)
        
        if np.isnan(reward) or np.isinf(reward):
            print(f"⚠️ Warning: Reward is NaN/Inf at step {self.ptr}, setting to 0")
            reward = 0.0
        
        if np.isnan(log_prob) or np.isinf(log_prob):
            print(f"⚠️ Warning: Log prob is NaN/Inf at step {self.ptr}, setting to -10")
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
        """
        使用 GAE (Generalized Advantage Estimation) 计算优势和回报
        
        Args:
            last_values: 最后一步的状态价值 (n_envs,)
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
        
        Returns:
            advantages: 优势估计 (实际存储长度,)
            returns: 回报估计 (实际存储长度,)
        """
        # 只使用实际存储的数据（前 ptr 个）
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
            
            # TD Error: delta = r + gamma * V(s') - V(s)
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        # Returns = Advantages + Values（只取实际存储的部分）
        returns = advantages + self.values[:actual_size]
        
        return advantages, returns
    
    def normalize_advantages(self, advantages: np.ndarray) -> np.ndarray:
        """
        优势归一化（关键！稳定 PPO 训练）
        
        Adv_normalized = (Adv - mean(Adv)) / (std(Adv) + eps)
        """
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    def get(self, batch_size: int, advantages: np.ndarray, returns: np.ndarray):
        """
        生成 mini-batch 用于更新
        
        Returns:
            generator of RolloutBufferSamples
        """
        # 归一化优势
        advantages = self.normalize_advantages(advantages)
        
        # 创建索引并打乱
        indices = np.random.permutation(self.ptr)
        
        # 生成 mini-batches
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
        """清空 buffer"""
        self.ptr = 0
    
    def __len__(self):
        return self.ptr


class PPOAgent:
    """
    PPO Agent for Portfolio Management
    
    核心算法流程：
    1. Collect Rollout (T steps)
    2. Compute GAE Advantages
    3. Update Network (K epochs, mini-batches)
    4. Clear Buffer
    
    使用 Dirichlet 分布处理投资组合权重
    """
    def __init__(self, network, config, device='cuda'):
        self.network = network
        self.config = config
        self.device = device
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.ppo.lr,
            eps=config.ppo.optimizer_eps,
            weight_decay=config.ppo.weight_decay
        )
        
        # 熵系数（可能衰减）
        self.entropy_coef = config.ppo.entropy_coef
        self.entropy_coef_final = config.ppo.entropy_coef_final
        self.entropy_decay = config.ppo.entropy_coef_decay
        
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, action_mask: np.ndarray, 
                      deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        选择动作
        
        Args:
            state: (S, T, F) 状态
            action_mask: (S,) 可交易掩码
            deterministic: 是否确定性策略
        
        Returns:
            action: (S,) 投资组合权重
            log_prob: 动作对数概率
            value: 状态价值
        """
        with torch.no_grad():
            # 输入数据清洗
            if np.isnan(state).any() or np.isinf(state).any():
                print("⚠️ Warning: Input state has NaN/Inf, cleaning...")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            
            # 前向传播
            alpha, dist, value = self.network(state_tensor, mask_tensor)
            
            # 采样动作（Dirichlet 分布）
            if deterministic:
                # 确定性：使用均值（alpha / alpha.sum()）
                action = alpha / alpha.sum(dim=-1, keepdim=True)
            else:
                # 随机采样
                action = dist.sample()
            
            # 确保动作有效
            action = torch.clamp(action, min=1e-10, max=1.0)
            action = action / action.sum(dim=-1, keepdim=True)  # 重新归一化
            
            # 计算对数概率
            log_prob = dist.log_prob(action)
            
            return (action.cpu().numpy()[0], 
                    log_prob.cpu().item(), 
                    value.cpu().item())
    
    def update(self, buffer: RolloutBuffer, advantages: np.ndarray, 
               returns: np.ndarray) -> Dict[str, float]:
        """
        PPO 网络更新
        
        Args:
            buffer: Rollout Buffer
            advantages: 优势估计
            returns: 回报估计
        
        Returns:
            loss_info: 训练指标字典
        """
        # 更新熵系数（线性衰减）
        if self.entropy_decay:
            progress = min(1.0, self.update_count / 1000)
            self.entropy_coef = self.entropy_coef * (1 - progress) + self.entropy_coef_final * progress
        
        # 收集所有 losses
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # 多轮更新（K epochs）
        for epoch in range(self.config.ppo.n_epochs):
            # 遍历 mini-batches
            for batch in buffer.get(self.config.ppo.mini_batch_size, advantages, returns):
                # 输入数据清洗
                observations = torch.nan_to_num(batch.observations, nan=0.0, posinf=1.0, neginf=-1.0)
                actions = torch.clamp(batch.actions, min=1e-10, max=1.0)
                
                # 计算新的 log_prob 和 value
                alpha, dist, new_log_probs, entropy, new_values = \
                    self.network.evaluate_actions(
                        observations, 
                        actions,
                        batch.action_masks
                    )
                
                # 检查数值有效性
                if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                    print("⚠️ Warning: new_log_probs has NaN/Inf, skipping batch")
                    continue
                
                if torch.isnan(batch.old_log_probs).any() or torch.isinf(batch.old_log_probs).any():
                    print("⚠️ Warning: old_log_probs has NaN/Inf, cleaning...")
                    old_log_probs_clean = torch.nan_to_num(batch.old_log_probs, nan=-10.0, posinf=0.0, neginf=-20.0)
                else:
                    old_log_probs_clean = batch.old_log_probs
                
                # ========== Policy Loss (PPO-Clip) ==========
                # 重要性采样比率：ratio = pi_new / pi_old
                ratio = torch.exp(new_log_probs - old_log_probs_clean)
                
                # 限制 ratio 范围，防止数值爆炸
                ratio = torch.clamp(ratio, min=0.1, max=10.0)
                
                # 裁剪后的替代损失
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - self.config.ppo.clip_range, 
                                   1 + self.config.ppo.clip_range) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 记录裁剪比例
                clip_fraction = (torch.abs(ratio - 1) > self.config.ppo.clip_range).float().mean()
                clip_fractions.append(clip_fraction.item())
                
                # ========== Value Loss ==========
                # 确保 value 和 returns 形状一致
                value_pred = new_values.squeeze()
                returns_batch = batch.returns
                
                # Returns 归一化（防止 value loss 过大）
                returns_mean = returns_batch.mean()
                returns_std = returns_batch.std() + 1e-8
                returns_normalized = (returns_batch - returns_mean) / returns_std
                
                # 对 value 也做相应的归一化（使其匹配 normalized returns）
                value_pred_normalized = (value_pred - returns_mean) / returns_std
                value_old_normalized = (batch.values - returns_mean) / returns_std
                
                # Value clipping（可选）
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
                
                # 检查 loss 有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Warning: Loss is NaN/Inf, skipping update. "
                          f"policy_loss={policy_loss.item()}, value_loss={value_loss.item()}, entropy_loss={entropy_loss.item()}")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（必须在 step 之前！）
                nn.utils.clip_grad_norm_(self.network.parameters(), 
                                        self.config.ppo.max_grad_norm)
                
                self.optimizer.step()
                
                # 记录 losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # 显存优化：删除不需要的 tensor 并清理缓存
                del observations, actions, alpha, dist, new_log_probs, entropy, new_values
                del ratio, surr1, surr2, policy_loss, value_loss, entropy_loss, loss
            
            # 每个 epoch 结束后清理显存
            torch.cuda.empty_cache()
        
        self.update_count += 1
        
        # 如果没有成功更新，返回空指标
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
        """从字典加载状态（用于增量学习）"""
        self.network.load_state_dict(state_dict['network'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.entropy_coef = state_dict.get('entropy_coef', self.config.ppo.entropy_coef)
