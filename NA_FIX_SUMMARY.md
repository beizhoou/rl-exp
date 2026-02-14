# NaN/Inf 紧急修复总结

## 问题诊断
- Actor 输出直接使用 Softmax，当概率接近 0 时 log(0) = -inf
- 输入数据含有 NaN/Inf 时传播到整个网络
- Advantage 归一化时除以 0 导致 NaN

## 修复内容

### 1. networks.py - 使用 Dirichlet 分布
```python
# Actor 输出浓度参数 alpha（必须 > 0）
alpha = F.softplus(logits) + 1e-8
dist = Dirichlet(alpha)  # 天然满足 sum=1, w>=0

# log_prob 计算安全
log_probs = dist.log_prob(action_clamped)
```

### 2. ppo_agent.py - 数值稳定
- 输入数据清洗：`torch.nan_to_num()`
- Ratio 限制范围：`torch.clamp(ratio, 0.1, 10.0)`
- Loss 有效性检查：`if torch.isnan(loss): continue`
- Advantage 归一化加 epsilon：`/(std + 1e-8)`

### 3. environment.py - 动作检查
```python
# Step 函数开头检查 NaN/Inf
if np.isnan(action).any() or np.isinf(action).any():
    print("🔥 CRITICAL: Action contains NaN/Inf!")
    action = np.ones(n_stocks) / n_stocks  # 回退到等权

# State 构建后检查
state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
```

## 运行命令
```bash
cd /root/autodl-tmp/exper_rl
python main_ppo.py --preprocessed-data processed_data.pkl --device cuda
```
