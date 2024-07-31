import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.utils.data as Data
import numpy as np

from server.PPO.models import ActorSoftmax, Critic
from server.PPO.memories import PGReplay


class Agent:
    def __init__(self, cfg) -> None:
        self.ppo_type = 'clip'  # clip or kl

        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)

        self.actor = ActorSoftmax(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs    # update policy for K epochs
        self.eps_clip = cfg.eps_clip    # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef    # entropy coefficient
        self.sample_count = 0
        self.train_batch_size = cfg.train_batch_size
        self.sgd_batch_size = cfg.sgd_batch_size

    def sample_action(self, state):
        # self.actor.train()
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.probs = probs.detach()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()

    @torch.no_grad()
    def predict_action(self, state):
        self.actor.eval()
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach()
        return action.detach().cpu().numpy().item()

    def update(self):
        # update policy every train_batch_size steps
        if self.sample_count % self.train_batch_size != 0:
            return
        # print("update policy")
        states, actions, rewards, dones, probs, log_probs = self.memory.sample()

        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32)    # shape:[train_batch_size,n_states]
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32).unsqueeze(dim=1)     # shape:[train_batch_size,1]
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32).unsqueeze(dim=1)     # shape:[train_batch_size,1]
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32).unsqueeze(dim=1)     # shape:[train_batch_size,1]
        probs = torch.cat(probs).to(self.device) # shape:[train_batch_size,n_actions]
        log_probs = torch.tensor(log_probs, device=self.device, dtype=torch.float32).unsqueeze(dim=1)   # shape:[train_batch_size,1]

        returns = self._compute_returns(rewards, dones) # shape:[train_batch_size,1]    
        torch_dataset = Data.TensorDataset(states, actions, probs, log_probs,returns)
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.sgd_batch_size, shuffle=True, drop_last=False)

        self.actor.train()
        self.critic.train()

        for _ in range(self.k_epochs):
            for batch_idx, (old_states, old_actions, old_probs, old_log_probs, returns) in enumerate(train_loader):
                # print('old_actions:', old_actions, ' old_probs: ',  old_probs)
                # compute advantages
                values = self.critic(old_states)     # detach to avoid backprop through the critic
                advantages = returns - values.detach()  # shape:[train_batch_size,1]

                # get action probabilities
                new_probs = self.actor(old_states) # shape:[train_batch_size,n_actions]
                dist = Categorical(new_probs)
                # get new action probabilities
                new_log_probs = dist.log_prob(old_actions.squeeze(dim=1)) # shape:[train_batch_size]
                # compute ratio (pi_theta / pi_theta__old):
                ratio = torch.exp(new_log_probs.unsqueeze(dim=1) - old_log_probs) # shape: [train_batch_size, 1]
                # compute surrogate loss
                surr1 = ratio * advantages # shape: [train_batch_size, 1]

                if self.ppo_type == 'clip':
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                    # compute actor loss
                    actor_loss = - (torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean())
                else:
                    raise NameError
                # compute critic loss
                critic_loss = nn.MSELoss()(returns, values) # shape: [train_batch_size, 1]
                # tot_loss = actor_loss + 0.5 * critic_loss
                print(f"actor loss: {actor_loss.item():.3f}, critic loss: {critic_loss.item():.3f}")
                # take gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                # tot_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def _compute_returns(self, rewards, dones):
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        return returns

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor.h5")
        torch.save(self.critic.state_dict(), f"{fpath}/critic.h5")

    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor.h5", map_location=self.device)
        critic_ckpt = torch.load(f"{fpath}/critic.h5", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)
