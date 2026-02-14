"""
高效特征编码器 - 解决Transformer batch size限制

核心改进:
1. 用Conv1d处理时间维度（无batch限制，更快）
2. 保持Cross-stock Attention捕捉截面效应
3. 支持任意batch size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class EfficientFeatureEncoder(nn.Module):
    """
    高效特征编码器：CNN处理时序 + Attention处理截面
    
    输入: (B, S, T, F) = (Batch, Stocks, Time, Features)
    输出: (B, S, d_model)
    """
    def __init__(self, n_features=40, n_stocks=471, d_model=128, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.n_stocks = n_stocks
        
        # 时间卷积：处理每个股票的时序特征
        # Conv1d: (B*S, F, T) -> (B*S, d_model, T')
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_features, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        
        # 时间池化：T维度压缩到1
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 跨股票注意力
        self.cross_stock = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, S, T, F)
        B, S, T, F = x.shape
        
        # 合并B和S维度用于Conv1d
        x = x.reshape(B * S, T, F)  # (B*S, T, F)
        x = x.permute(0, 2, 1)  # (B*S, F, T) - Conv1d需要channels在前
        
        # 时间卷积 (没有batch限制！)
        x = self.temporal_conv(x)  # (B*S, d_model, T)
        
        # 时间池化
        x = self.temporal_pool(x).squeeze(-1)  # (B*S, d_model)
        
        # 恢复形状
        x = x.reshape(B, S, self.d_model)  # (B, S, d_model)
        x = self.norm1(x)
        
        # 跨股票注意力 (B=2048没问题)
        attn_out, _ = self.cross_stock(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        return x


class Actor(nn.Module):
    """PPO Actor - Dirichlet分布"""
    def __init__(self, n_stocks=471, encoder=None, temperature=1.0):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model if encoder else 128
        self.temperature = temperature
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)
        )
        
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
    def forward(self, state, action_mask=None):
        if state.dim() == 4:
            features = self.encoder(state)
        else:
            features = state
        
        logits = self.policy_head(features).squeeze(-1)
        
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        alpha = F.softplus(logits) + 1e-8
        dist = Dirichlet(alpha)
        
        return alpha, dist


class Critic(nn.Module):
    """PPO Critic - 状态价值估计"""
    def __init__(self, n_stocks=471, encoder=None):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model if encoder else 128
        
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)
        )
        
    def forward(self, state):
        if state.dim() == 4:
            features = self.encoder(state)
        else:
            features = state
        
        # 聚合所有股票信息
        feat_mean = features.mean(dim=1)  # (B, d_model)
        feat_max = features.max(dim=1)[0]  # (B, d_model)
        aggregated = torch.cat([feat_mean, feat_max], dim=-1)
        
        value = self.value_head(aggregated)
        return value


class ActorCriticNetwork(nn.Module):
    """组合Actor和Critic网络"""
    def __init__(self, n_stocks=471, n_features=40, lookback=20, 
                 d_model=128, temperature=1.0):
        super().__init__()
        
        self.encoder = EfficientFeatureEncoder(n_features, n_stocks, d_model)
        self.actor = Actor(n_stocks, self.encoder, temperature)
        self.critic = Critic(n_stocks, self.encoder)
        
    def forward(self, state, action_mask=None):
        """前向传播返回alpha, dist, value"""
        features = self.encoder(state)
        alpha, dist = self.actor(features, action_mask)
        value = self.critic(features)
        return alpha, dist, value
    
    def evaluate_actions(self, state, action, action_mask):
        """评估动作（用于PPO更新）"""
        features = self.encoder(state)
        
        # Actor
        alpha, dist = self.actor(features, action_mask)
        log_probs = dist.log_prob(action.clamp(min=1e-10))
        entropy = dist.entropy()
        
        # Critic
        value = self.critic(features)
        
        return alpha, dist, log_probs, entropy, value


def create_networks(n_stocks, n_features, lookback, d_model=128, 
                   temperature=1.0, device='cuda'):
    """创建网络"""
    network = ActorCriticNetwork(
        n_stocks=n_stocks,
        n_features=n_features,
        lookback=lookback,
        d_model=d_model,
        temperature=temperature
    ).to(device)
    
    return network


# 保持向后兼容
HeterogeneousFeatureEncoder = EfficientFeatureEncoder
