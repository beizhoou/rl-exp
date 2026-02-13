import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Tuple

class PortfolioSAC:
    def __init__(self, actor, critic1, critic2, target_critic1, target_critic2, 
                 config, device):
        self.device = device
        self.config = config
        
        self.actor = actor.to(device)
        self.critic1 = critic1.to(device)
        self.critic2 = critic2.to(device)
        self.target_critic1 = target_critic1.to(device)
        self.target_critic2 = target_critic2.to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.sac.lr, eps=1e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.sac.lr, eps=1e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.sac.lr, eps=1e-4)
        
        self.target_entropy = -config.data.n_stocks * 0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.sac.lr)
        
        self.use_amp = config.sac.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.gamma = config.sac.gamma
        self.tau = config.sac.tau
        self.max_grad_norm = config.sac.max_grad_norm
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    weights, _ = self.actor(state, deterministic)
            else:
                weights, _ = self.actor(state, deterministic)
        return weights.cpu().numpy()[0]
    
    def update(self, batch: Tuple) -> Dict[str, float]:
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Critic update
        with autocast(enabled=self.use_amp):
            with torch.no_grad():
                next_weights, _ = self.actor(next_states)
                next_q1 = self.target_critic1(next_states, next_weights)
                next_q2 = self.target_critic2(next_states, next_weights)
                next_q = torch.min(next_q1, next_q2)
                entropy = -torch.sum(next_weights * torch.log(next_weights + 1e-8), dim=1, keepdim=True)
                target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * entropy)
            
            current_q1 = self.critic1(states, actions)
            current_q2 = self.critic2(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic1_optimizer)
            self.scaler.unscale_(self.critic2_optimizer)
        else:
            critic_loss.backward()
            
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        
        if self.use_amp:
            self.scaler.step(self.critic1_optimizer)
            self.scaler.step(self.critic2_optimizer)
        else:
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
        
        # Actor update
        actor_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        
        if self.critic1_optimizer.state_dict().get('state', {}).get('step', 0) % 2 == 0:
            with autocast(enabled=self.use_amp):
                new_weights, _ = self.actor(states)
                q1_new = self.critic1(states, new_weights)
                q2_new = self.critic2(states, new_weights)
                q_new = torch.min(q1_new, q2_new)
                entropy = -torch.sum(new_weights * torch.log(new_weights + 1e-8), dim=1, keepdim=True)
                actor_loss = -(q_new - self.alpha.detach() * entropy).mean()
                alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
            
            self.actor_optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
            else:
                actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            if self.use_amp:
                self.scaler.step(self.actor_optimizer)
            else:
                self.actor_optimizer.step()
            
            self.alpha_optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(alpha_loss).backward()
                self.scaler.step(self.alpha_optimizer)
            else:
                alpha_loss.backward()
                self.alpha_optimizer.step()
        
        # Soft update
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        if self.use_amp:
            self.scaler.update()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0,
            'alpha': self.alpha.item(),
            'entropy': entropy.mean().item() if isinstance(entropy, torch.Tensor) else 0.0
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'alpha': self.log_alpha
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.log_alpha = checkpoint['alpha']