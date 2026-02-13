import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HeterogeneousFeatureEncoder(nn.Module):
    """
    异构特征编码器：处理40维混合特征（价量/基本面/情绪/研报）
    使用分组注意力：不同特征组使用不同注意力头
    """
    def __init__(self, n_features=40, n_stocks=471, d_model=128, nhead=4):
        super().__init__()
        self.d_model = d_model
        self.n_stocks = n_stocks
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 分组编码：根据特征来源分组处理（可选的高级特性）
        # 这里使用标准Transformer Encoder
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
        
        # 时序编码
        x = x.view(B*S, T, F)
        x = self.input_proj(x)
        x = self.temporal_encoder(x)
        x = x[:, -1, :].view(B, S, self.d_model)  # 取最后时刻
        x = self.norm1(x)
        
        # 跨股票注意力（捕捉行业轮动等截面效应）
        attn_out, _ = self.cross_stock(x, x, x)
        x = x + attn_out
        x = self.norm2(x)
        
        return x

class Actor(nn.Module):
    """策略网络"""
    def __init__(self, n_stocks=471, encoder=None):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.d_model if encoder else 128
        
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model//2, 1)
        )
        
        # 小的初始化，避免初期过度集中
        nn.init.xavier_uniform_(self.policy_head[-1].weight, gain=0.01)
        
    def forward(self, state, deterministic=False):
        if state.dim() == 3:
            state = state.unsqueeze(0)
            
        features = self.encoder(state)
        logits = self.policy_head(features).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        
        # 训练时添加Dirichlet噪声保持探索
        if not deterministic and self.training:
            noise = torch.distributions.Dirichlet(torch.ones_like(weights)*0.1).sample()
            weights = 0.9 * weights + 0.1 * noise
            weights = weights / weights.sum(dim=-1, keepdim=True)
            
        return weights, logits

class Critic(nn.Module):
    """价值网络（Double-Q）"""
    def __init__(self, n_stocks=471, encoder=None):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.d_model if encoder else 128
        
        self.q_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model),  # +1 for concentration feature
            nn.ReLU(),
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
        
    def forward(self, state, action):
        features = self.encoder(state)  # (B, S, D)
        
        # 组合特征：加权特征 + 集中度
        portfolio_feat = torch.sum(features * action.unsqueeze(-1), dim=1)
        hhi = torch.sum(action ** 2, dim=1, keepdim=True)  # Herfindahl指数
        
        x = torch.cat([portfolio_feat, hhi], dim=-1)
        return self.q_head(x)

def create_networks(n_stocks, n_features, lookback, d_model, device):
    """工厂函数"""
    encoder = HeterogeneousFeatureEncoder(n_features, n_stocks, d_model).to(device)
    actor = Actor(n_stocks, encoder).to(device)
    critic1 = Critic(n_stocks, encoder).to(device)
    critic2 = Critic(n_stocks, encoder).to(device)
    
    target_critic1 = Critic(n_stocks, encoder).to(device)
    target_critic2 = Critic(n_stocks, encoder).to(device)
    
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    
    return actor, critic1, critic2, target_critic1, target_critic2