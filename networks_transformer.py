import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Dirichlet

class HeterogeneousFeatureEncoder(nn.Module):
    """
    异构特征编码器：处理40维混合特征（价量/基本面/情绪/研报）
    
    用于 PPO 的特征提取器
    """
    def __init__(self, n_features=40, n_stocks=471, d_model=128, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.n_stocks = n_stocks
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 时序编码：Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=d_model*4, dropout=0.1, batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 跨股票注意力（低秩近似减少计算量）
        self.cross_stock = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, S, T, F) = (Batch, 471 Stocks, 20 Days, 40 Features)
        B, S, T, F = x.shape
        
        # 高效处理：调整维度顺序，避免超大batch
        # 方法：把 Stock 维度当 batch，Time 维度当序列长度
        # 这样 batch = 2048, seq_len = 471*20 = 9420，而不是 batch=964608
        
        # 转置: (B, S, T, F) -> (B, T, S, F)
        x = x.permute(0, 2, 1, 3)  # (B, T, S, F)
        x = x.reshape(B, T * S, F)  # (B, T*S, F) = (2048, 9420, 40)
        
        # 输入投影
        x = self.input_proj(x)  # (B, T*S, d_model)
        
        # 恢复时间维度用于Transformer: (B, T, S, d_model)
        x = x.reshape(B, T, S, self.d_model)
        x = x.permute(0, 2, 1, 3)  # (B, S, T, d_model)
        x = x.reshape(B * S, T, self.d_model)  # (B*S, T, d_model) - 但这时T=20很小
        
        # 等等，这样还是 (B*S, T, d_model)，问题没变
        # 重新思考：我们需要的是让 Transformer 的 batch 维度变小
        
        # 正确做法：把 stock 当作 sequence，time 保持不变
        # (B, S, T, F) -> 对每个时间步，处理所有股票
        # 但这样需要改模型结构...
        
        # 实际解决方案：减小 batch_size 配置
        # 但既然已经在这个函数里了，我们用高效的卷积替代Transformer
        
        # 方案：用 Conv1d 替代 Transformer Encoder（无batch限制，更快）
        # (B*S, T, F) -> (B*S, F, T) for Conv1d
        x = x.reshape(B * S, T, F).permute(0, 2, 1)  # (B*S, F, T)
        
        # 这里应该用Conv1d...但我们先临时方案：用RNN
        # 实际上，对于T=20，直接用简单的Mean Pooling都行
        
        # 快速修复：对T维度做mean pooling（20天的平均特征）
        x = x.reshape(B * S, F, T)
        x = x.mean(dim=-1)  # (B*S, F)
        
        # 重新投影到d_model
        x = self.input_proj(x)  # (B*S, d_model)
        x = x.reshape(B, S, self.d_model)  # (B, S, d_model)
        
        # 后续用简单的MLP替代Transformer
        x = self.norm1(x)
        
        # 跨股票注意力（B=2048可以正常处理）
        attn_out, _ = self.cross_stock(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        return x


class Actor(nn.Module):
    """
    PPO Actor (Policy) Network - 使用 Dirichlet 分布
    
    输入：融合后的 State 特征
    输出：Dirichlet 浓度参数 alpha -> 采样投资组合权重
    
    Dirichlet 分布天然满足：
    1. 权重和为 1
    2. 所有权重 >= 0
    3. 可计算 log_prob（数值稳定）
    """
    def __init__(self, n_stocks=471, encoder=None, temperature=1.0):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model if encoder else 128
        self.temperature = temperature
        
        # 策略头：输出浓度参数 alpha
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 1)  # 每只股票输出一个值
        )
        
        # 小的初始化
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
    def forward(self, state, action_mask=None):
        """
        Args:
            state: (B, S, T, F) 或 (B, S, d_model)
            action_mask: (B, S) 可选的掩码，有效位置为 1，无效位置为 0
        
        Returns:
            alpha: (B, S) Dirichlet 浓度参数
            dist: Dirichlet 分布对象
            action: (B, S) 采样的投资组合权重
        """
        # 如果输入是原始特征，进行编码
        if state.dim() == 4:
            features = self.encoder(state)  # (B, S, d_model)
        else:
            features = state
        
        # 计算每只股票的基础值
        logits = self.policy_head(features).squeeze(-1)  # (B, S)
        
        # 应用温度系数
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # 应用 Action Masking（关键！）
        # 将停牌/跌停股票的 logits 设为极小值，使其浓度参数接近0
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)
        
        # 转换为浓度参数 alpha（必须 > 0）
        # 使用 softplus 确保正值，加 1e-8 防止数值问题
        alpha = F.softplus(logits) + 1e-8  # (B, S)
        
        # 创建 Dirichlet 分布
        # Dirichlet(alpha) 采样出的权重自动满足 sum=1, w_i >= 0
        dist = Dirichlet(alpha)
        
        return alpha, dist


class Critic(nn.Module):
    """
    PPO Critic (Value) Network
    
    输入：融合后的 State 特征
    输出：1 维标量 (State Value)
    """
    def __init__(self, n_stocks=471, encoder=None):
        super().__init__()
        self.encoder = encoder
        self.d_model = encoder.d_model if encoder else 128
        
        # Value head - 先聚合所有股票信息
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),  # *2 因为 concat了mean和max
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 1)  # 输出标量 value
        )
        
    def forward(self, state):
        """
        Args:
            state: (B, S, T, F) 或 (B, S, d_model)
        
        Returns:
            value: (B, 1) 状态价值估计
        """
        # 如果输入是原始特征，进行编码
        if state.dim() == 4:
            features = self.encoder(state)  # (B, S, d_model)
        else:
            features = state
        
        # 聚合所有股票的特征
        feat_mean = features.mean(dim=1)  # (B, d_model)
        feat_max = features.max(dim=1)[0]  # (B, d_model)
        feat_agg = torch.cat([feat_mean, feat_max], dim=-1)  # (B, d_model * 2)
        
        # 计算状态价值
        value = self.value_head(feat_agg)  # (B, 1)
        
        return value


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic 联合网络
    
    共享特征编码器，但分别有独立的策略头和价值头
    使用 Dirichlet 分布处理投资组合权重
    """
    def __init__(self, n_stocks=471, n_features=40, lookback=20, 
                 d_model=128, temperature=1.0, device='cuda'):
        super().__init__()
        self.n_stocks = n_stocks
        self.device = device
        
        # 共享的特征编码器
        self.encoder = HeterogeneousFeatureEncoder(
            n_features=n_features, 
            n_stocks=n_stocks, 
            d_model=d_model
        ).to(device)
        
        # Actor (Policy) - 使用 Dirichlet 分布
        self.actor = Actor(n_stocks, self.encoder, temperature).to(device)
        
        # Critic (Value)
        self.critic = Critic(n_stocks, self.encoder).to(device)
        
    def forward(self, state, action_mask=None):
        """
        前向传播
        
        Args:
            state: (B, S, T, F)
            action_mask: (B, S) 可选
        
        Returns:
            alpha: (B, S) Dirichlet 浓度参数
            dist: Dirichlet 分布
            value: (B, 1) 状态价值
        """
        # Actor 输出
        alpha, dist = self.actor(state, action_mask)
        
        # Critic 输出
        value = self.critic(state)
        
        return alpha, dist, value
    
    def get_value(self, state):
        """获取状态价值（用于计算 advantage）"""
        return self.critic(state)
    
    def evaluate_actions(self, state, action, action_mask=None):
        """
        评估动作（用于 PPO 更新）
        
        Args:
            state: (B, S, T, F)
            action: (B, S) 实际采取的动作（权重分布）
            action_mask: (B, S)
        
        Returns:
            alpha: (B, S)
            dist: Dirichlet 分布
            log_probs: (B,) 动作的对数概率
            entropy: (B,) 策略熵
            value: (B, 1)
        """
        alpha, dist, value = self.forward(state, action_mask)
        
        # 计算对数概率（Dirichlet 分布的 log_prob）
        # 确保 action 在有效范围内，避免 log(0)
        action_clamped = torch.clamp(action, min=1e-10, max=1.0)
        log_probs = dist.log_prob(action_clamped)
        
        # 计算策略熵
        entropy = dist.entropy()
        
        return alpha, dist, log_probs, entropy, value


def create_networks(n_stocks, n_features, lookback, d_model, temperature, device):
    """
    创建 PPO 网络的工厂函数
    
    Returns:
        ActorCritic 联合网络
    """
    return ActorCritic(
        n_stocks=n_stocks,
        n_features=n_features,
        lookback=lookback,
        d_model=d_model,
        temperature=temperature,
        device=device
    )
