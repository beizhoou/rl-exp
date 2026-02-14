"""
PPO Networks V2 - 支持现金仓位和改进策略分布

核心改进:
1. 支持现金仓位 (n_stocks + 1)
2. 可选策略分布: Dirichlet 或 Softmax + 高斯噪声
3. Top-K 截断支持（降低优化难度）
4. 更稳定的数值计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal
import math


class EfficientFeatureEncoder(nn.Module):
    """
    高效特征编码器：CNN处理时序 + Attention处理截面
    V2: 支持现金仓位特征处理
    """
    def __init__(self, n_features=40, n_stocks=471, d_model=128, nhead=4, enable_cash=True):
        super().__init__()
        self.d_model = d_model
        self.n_stocks = n_stocks
        self.enable_cash = enable_cash
        self.n_assets = n_stocks + (1 if enable_cash else 0)
        
        # 时间卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_features, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
        )
        
        # 时间池化
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 跨股票注意力（V2: 包含现金）
        self.cross_stock = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # V2: 现金特征编码（学习现金的表示）
        if enable_cash:
            # 现金是一个可学习的嵌入向量
            self.cash_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)
        
    def forward(self, x):
        """
        V2: 处理包含现金的状态
        x: (B, n_assets, T, F) 其中n_assets = n_stocks + 1 (cash)
        """
        B, S, T, F = x.shape
        
        # 分离股票和现金（如果启用）
        if self.enable_cash and S == self.n_assets:
            stock_features = x[:, :self.n_stocks, :, :]  # (B, n_stocks, T, F)
            # 现金特征x[:, self.n_stocks:, :, :] 应该是全0或特殊值
        else:
            stock_features = x
        
        S_stock = stock_features.shape[1]
        
        # 处理股票特征
        # 合并B和S维度用于Conv1d
        x_stock = stock_features.reshape(B * S_stock, T, F)
        x_stock = x_stock.permute(0, 2, 1)  # (B*S, F, T)
        
        # 时间卷积
        x_stock = self.temporal_conv(x_stock)  # (B*S, d_model, T)
        x_stock = self.temporal_pool(x_stock).squeeze(-1)  # (B*S, d_model)
        
        # 恢复形状
        x_stock = x_stock.reshape(B, S_stock, self.d_model)  # (B, n_stocks, d_model)
        
        # V2: 添加现金嵌入
        if self.enable_cash and S == self.n_assets:
            cash_embed = self.cash_embedding.expand(B, 1, self.d_model)  # (B, 1, d_model)
            x = torch.cat([x_stock, cash_embed], dim=1)  # (B, n_assets, d_model)
        else:
            x = x_stock
        
        x = self.norm1(x)
        
        # 跨资产注意力
        attn_out, _ = self.cross_stock(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        return x  # (B, n_assets, d_model)


class ActorDirichlet(nn.Module):
    """
    V2: Dirichlet策略（原始版本，用于对比）
    """
    def __init__(self, n_stocks=471, d_model=128, temperature=1.0, enable_cash=True):
        super().__init__()
        self.n_stocks = n_stocks
        self.enable_cash = enable_cash
        self.n_assets = n_stocks + (1 if enable_cash else 0)
        self.temperature = temperature
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
    def forward(self, features, action_mask=None):
        """
        features: (B, n_assets, d_model)
        action_mask: (B, n_assets) 可选
        """
        logits = self.policy_head(features).squeeze(-1)  # (B, n_assets)
        
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Action Masking
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        # 转换为浓度参数alpha
        alpha = F.softplus(logits) + 1e-8  # (B, n_assets)
        
        # 创建Dirichlet分布
        dist = Dirichlet(alpha)
        
        return alpha, dist
    
    def sample(self, dist, deterministic=False):
        """采样动作"""
        if deterministic:
            # 使用均值作为确定性动作
            alpha = dist.concentration
            action = alpha / alpha.sum(dim=-1, keepdim=True)
        else:
            action = dist.sample()
        
        # 确保有效
        action = torch.clamp(action, min=1e-10, max=1.0)
        action = action / action.sum(dim=-1, keepdim=True)
        
        return action
    
    def log_prob(self, dist, action):
        """计算对数概率"""
        action_clamped = torch.clamp(action, min=1e-10, max=1.0)
        return dist.log_prob(action_clamped)
    
    def entropy(self, dist):
        """计算熵"""
        return dist.entropy()


class ActorSoftmaxGaussian(nn.Module):
    """
    V2: Softmax + 高斯噪声策略（更稳定，推荐）
    
    改进点：
    1. 输出基础权重mu
    2. 在logits层添加高斯噪声进行探索
    3. 更容易产生稀疏权重（Top-K）
    """
    def __init__(self, n_stocks=471, d_model=128, temperature=1.0, enable_cash=True, 
                 noise_std=0.1, top_k=None):
        super().__init__()
        self.n_stocks = n_stocks
        self.enable_cash = enable_cash
        self.n_assets = n_stocks + (1 if enable_cash else 0)
        self.temperature = temperature
        self.noise_std = noise_std
        self.top_k = top_k  # Top-K截断，如10表示只保留前10只股票
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )
        
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
        # 噪声参数（可学习或固定）
        self.log_noise_std = nn.Parameter(torch.ones(1) * math.log(noise_std))
        
    def forward(self, features, action_mask=None):
        """
        features: (B, n_assets, d_model)
        action_mask: (B, n_assets)
        
        Returns:
            mu: 基础权重 (B, n_assets)
            dist: 用于采样的分布
        """
        logits = self.policy_head(features).squeeze(-1)  # (B, n_assets)
        
        # 应用温度
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Action Masking
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        # 计算基础概率（确定性部分）
        mu = F.softmax(logits, dim=-1)  # (B, n_assets)
        
        # 创建高斯分布用于采样（在logits空间添加噪声）
        noise_std = torch.exp(self.log_noise_std)
        
        # 我们返回mu和std，实际采样在sample方法中
        return mu, noise_std, logits
    
    def sample(self, mu, noise_std, logits, deterministic=False, action_mask=None):
        """
        采样动作
        
        V2改进:
        - deterministic=False: 在logits上添加高斯噪声，然后softmax
        - deterministic=True: 直接使用mu
        """
        if deterministic:
            action = mu
        else:
            # 添加高斯噪声到logits
            noise = torch.randn_like(logits) * noise_std
            noisy_logits = logits + noise
            
            # 重新应用mask
            if action_mask is not None:
                noisy_logits = noisy_logits.masked_fill(action_mask == 0, -1e9)
            
            action = F.softmax(noisy_logits, dim=-1)
        
        # V2: Top-K截断（如果启用）
        if self.top_k is not None and self.top_k < self.n_assets:
            # 保留Top-K，其余设为0
            top_k_values, top_k_indices = torch.topk(action, k=self.top_k, dim=-1)
            action_new = torch.zeros_like(action)
            action_new.scatter_(1, top_k_indices, top_k_values)
            action = action_new
            # 重新归一化
            action = action / (action.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 确保有效
        action = torch.clamp(action, min=1e-10, max=1.0)
        action = action / action.sum(dim=-1, keepdim=True)
        
        return action
    
    def log_prob(self, action, mu, noise_std):
        """
        计算对数概率（近似）
        对于Softmax+Gaussian，使用高斯分布近似
        """
        # 将对数概率近似为高斯分布
        # 这是一种简化，实际应该用更复杂的计算
        dist = Normal(mu, noise_std)
        log_probs = dist.log_prob(action).sum(dim=-1)
        return log_probs
    
    def entropy(self, noise_std):
        """计算策略熵（基于噪声标准差）"""
        # 噪声越大，熵越高
        return 0.5 * math.log(2 * math.pi * math.e) + torch.log(noise_std)


class Critic(nn.Module):
    """
    V2: Critic支持现金仓位
    """
    def __init__(self, n_stocks=471, d_model=128, enable_cash=True):
        super().__init__()
        self.n_stocks = n_stocks
        self.enable_cash = enable_cash
        self.n_assets = n_stocks + (1 if enable_cash else 0)
        
        self.value_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, features):
        """
        features: (B, n_assets, d_model)
        """
        # 聚合所有资产信息
        feat_mean = features.mean(dim=1)  # (B, d_model)
        feat_max = features.max(dim=1)[0]  # (B, d_model)
        aggregated = torch.cat([feat_mean, feat_max], dim=-1)  # (B, d_model * 2)
        
        value = self.value_head(aggregated)
        return value


class ActorCriticNetworkV2(nn.Module):
    """
    V2: Actor-Critic联合网络
    支持两种策略分布: Dirichlet 或 Softmax+Gaussian
    支持现金仓位
    """
    def __init__(self, n_stocks=471, n_features=40, lookback=20, 
                 d_model=128, temperature=1.0, device='cuda',
                 enable_cash=True, policy_type='softmax_gaussian', top_k=None):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_assets = n_stocks + (1 if enable_cash else 0)
        self.device = device
        self.enable_cash = enable_cash
        self.policy_type = policy_type
        
        # 共享的特征编码器
        self.encoder = EfficientFeatureEncoder(
            n_features=n_features,
            n_stocks=n_stocks,
            d_model=d_model,
            enable_cash=enable_cash
        ).to(device)
        
        # 根据策略类型选择Actor
        if policy_type == 'dirichlet':
            self.actor = ActorDirichlet(
                n_stocks=n_stocks,
                d_model=d_model,
                temperature=temperature,
                enable_cash=enable_cash
            ).to(device)
        elif policy_type == 'softmax_gaussian':
            self.actor = ActorSoftmaxGaussian(
                n_stocks=n_stocks,
                d_model=d_model,
                temperature=temperature,
                enable_cash=enable_cash,
                top_k=top_k
            ).to(device)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        
        # Critic
        self.critic = Critic(
            n_stocks=n_stocks,
            d_model=d_model,
            enable_cash=enable_cash
        ).to(device)
        
    def forward(self, state, action_mask=None):
        """
        前向传播
        
        Args:
            state: (B, n_assets, T, F)
            action_mask: (B, n_assets)
        
        Returns:
            根据policy_type返回不同格式
        """
        features = self.encoder(state)
        
        if self.policy_type == 'dirichlet':
            alpha, dist = self.actor(features, action_mask)
            value = self.critic(features)
            return alpha, dist, value
        else:  # softmax_gaussian
            mu, noise_std, logits = self.actor(features, action_mask)
            value = self.critic(features)
            return mu, noise_std, logits, value
    
    def get_value(self, state):
        """获取状态价值"""
        features = self.encoder(state)
        return self.critic(features)
    
    def select_action(self, state, action_mask=None, deterministic=False):
        """
        选择动作（统一接口）
        
        Returns:
            action: (B, n_assets)
            log_prob: (B,)
            value: (B, 1)
        """
        if self.policy_type == 'dirichlet':
            features = self.encoder(state)
            alpha, dist = self.actor(features, action_mask)
            action = self.actor.sample(dist, deterministic)
            log_prob = self.actor.log_prob(dist, action)
            entropy = self.actor.entropy(dist)
            value = self.critic(features)
            return action, log_prob, value, entropy
        else:  # softmax_gaussian
            features = self.encoder(state)
            mu, noise_std, logits = self.actor(features, action_mask)
            action = self.actor.sample(mu, noise_std, logits, deterministic, action_mask)
            log_prob = self.actor.log_prob(action, mu, noise_std)
            entropy = self.actor.entropy(noise_std)
            value = self.critic(features)
            return action, log_prob, value, entropy
    
    def evaluate_actions(self, state, action, action_mask=None):
        """
        评估动作（用于PPO更新）
        
        Returns:
            log_probs: (B,)
            entropy: (B,)
            value: (B, 1)
        """
        if self.policy_type == 'dirichlet':
            features = self.encoder(state)
            alpha, dist = self.actor(features, action_mask)
            log_probs = self.actor.log_prob(dist, action)
            entropy = self.actor.entropy(dist)
            value = self.critic(features)
            return alpha, dist, log_probs, entropy, value
        else:  # softmax_gaussian
            features = self.encoder(state)
            mu, noise_std, logits = self.actor(features, action_mask)
            log_probs = self.actor.log_prob(action, mu, noise_std)
            # 扩展entropy到batch维度
            entropy = self.actor.entropy(noise_std).expand(mu.shape[0])
            value = self.critic(features)
            return mu, None, log_probs, entropy, value


def create_networks_v2(n_stocks, n_features, lookback, d_model=128, 
                       temperature=1.0, device='cuda', enable_cash=True,
                       policy_type='softmax_gaussian', top_k=None):
    """
    创建V2网络的工厂函数
    
    Args:
        n_stocks: 股票数量
        n_features: 特征维度
        lookback: 回看窗口
        d_model: 模型维度
        temperature: 温度系数
        device: 计算设备
        enable_cash: 是否启用现金仓位
        policy_type: 'dirichlet' 或 'softmax_gaussian'
        top_k: Top-K截断数量，None表示不截断
    """
    return ActorCriticNetworkV2(
        n_stocks=n_stocks,
        n_features=n_features,
        lookback=lookback,
        d_model=d_model,
        temperature=temperature,
        device=device,
        enable_cash=enable_cash,
        policy_type=policy_type,
        top_k=top_k
    )
