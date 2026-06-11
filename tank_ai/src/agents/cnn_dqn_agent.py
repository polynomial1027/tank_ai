from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tank_ai.src.models.cnn_dqn import CNNDQN
from tank_ai.src.utils.replay_buffer import ReplayBuffer


class CNNDQNAgent:
    def __init__(
        self,
        input_channels: int = 6,
        output_size: int = 6,
        rows: int = 15,
        cols: int = 20,
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        batch_size: int = 128,
        memory_size: int = 80_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_every: int = 20,
        device: str | None = None,
    ) -> None:
        self.output_size = output_size
        self.rows = rows
        self.cols = cols
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_every = target_update_every
        self.device = torch.device(device or self._default_device())
        self.policy_net = CNNDQN(input_channels, output_size, rows, cols).to(self.device)
        self.target_net = CNNDQN(input_channels, output_size, rows, cols).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(memory_size)
        self.training_steps = 0
        self.episode_count = 0

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_t.dim() == 3:
            state_t = state_t.unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(state_t)
            action = torch.argmax(q, dim=1).item()
        self.policy_net.train()
        return int(action)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (~dones_t)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1
        return float(loss.item())

    def end_episode(self) -> None:
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        if self.episode_count % self.target_update_every == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "episode_count": self.episode_count,
            "rows": self.rows,
            "cols": self.cols,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.training_steps = ckpt.get("training_steps", 0)
        self.episode_count = ckpt.get("episode_count", 0)
        self.policy_net.train()
        self.target_net.eval()

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
