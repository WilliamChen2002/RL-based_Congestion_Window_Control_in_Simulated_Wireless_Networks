"""
agent.py — DQN Agent + DDPG Agent

API 約定：
  State  : [cwnd, throughput, aoi, loss_rate, queue]（正規化，共 5 維，各值 0.0~1.0）
           無 AoI 版本為 4 維：[cwnd, throughput, loss_rate, queue]
  Reward : 由環境計算後回傳（agent 不自己算）

  DQN：
    Action : 0=減少 / 1=不變 / 2=增加 cwnd（離散）

  DDPG：
    Action : 目標 cwnd 值，範圍 [1.0, 100.0]（連續）
"""

import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── 共用設定 ──
GAMMA = 0.99
BATCH_SIZE = 64

# ══════════════════════════════════════════════════════
#  DQN
# ══════════════════════════════════════════════════════

DQN_STATE_DIM = 5  # 預設：[cwnd, throughput, aoi, loss_rate, queue]
DQN_ACTION_DIM = 3
DQN_LR = 1e-3
DQN_BUFFER_SIZE = 10_000
DQN_LEARNING_STARTS = 500
DQN_TARGET_UPDATE_FREQ = 500
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 0.99
DQN_GRAD_CLIP = 1.0


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = DQN_STATE_DIM,
        action_dim: int = DQN_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = DQN_STATE_DIM,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = state_dim

        self.q_net = QNetwork(state_dim).to(self.device)
        self.target_net = QNetwork(state_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.buffer = ReplayBuffer(DQN_BUFFER_SIZE)

        self.eps = DQN_EPS_START
        self._total_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randint(0, DQN_ACTION_DIM - 1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax().item())

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.buffer) < DQN_LEARNING_STARTS:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t = torch.tensor(states).to(self.device)
        actions_t = (
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        )
        rewards_t = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1).values
            target = rewards_t + GAMMA * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=DQN_GRAD_CLIP)
        self.optimizer.step()

        self._total_steps += 1
        if self._total_steps % DQN_TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        self.eps = max(DQN_EPS_END, self.eps * DQN_EPS_DECAY)

    def save(self, path: str = "dqn_agent.pth") -> None:
        torch.save(self.q_net.state_dict(), path)
        print(f"模型已儲存：{path}")

    def load(self, path: str = "dqn_agent.pth") -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"模型已載入：{path}")


# ══════════════════════════════════════════════════════
#  DDPG
# ══════════════════════════════════════════════════════

DDPG_STATE_DIM = 5  # 預設：[cwnd, throughput, aoi, loss_rate, queue]
DDPG_ACTION_DIM = 1
DDPG_ACTION_LOW = 1.0
DDPG_ACTION_HIGH = 100.0
DDPG_LR_ACTOR = 1e-4
DDPG_LR_CRITIC = 1e-3
DDPG_BUFFER_SIZE = 100_000
DDPG_LEARNING_STARTS = 2000
DDPG_TAU = 0.005
DDPG_GRAD_CLIP = 1.0
DDPG_NOISE_SIGMA_START = 3.0
DDPG_NOISE_SIGMA_END = 0.3
DDPG_NOISE_DECAY = 0.99


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int = DDPG_STATE_DIM,
        action_low: float = DDPG_ACTION_LOW,
        action_high: float = DDPG_ACTION_HIGH,
    ) -> None:
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.net(state)
        return x * (self.action_high - self.action_low) + self.action_low


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int = DDPG_STATE_DIM,
        action_dim: int = DDPG_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class DDPGAgent:
    def __init__(
        self,
        state_dim: int = DDPG_STATE_DIM,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = state_dim

        self.actor = Actor(state_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.eval()

        self.critic = Critic(state_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=DDPG_LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=DDPG_LR_CRITIC)

        self.buffer = ReplayBuffer(DDPG_BUFFER_SIZE)
        self.noise_sigma = DDPG_NOISE_SIGMA_START

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if explore:
            noise = np.random.normal(0, self.noise_sigma, size=action.shape)
            action = action + noise

        return np.clip(action, DDPG_ACTION_LOW, DDPG_ACTION_HIGH).astype(np.float32)

    def store(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> dict | None:
        if len(self.buffer) < DDPG_LEARNING_STARTS:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            next_q = self.critic_target(next_states_t, next_actions)
            target_q = rewards_t + GAMMA * next_q * (1 - dones_t)

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=DDPG_GRAD_CLIP)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        for p_online, p_target in zip(
            online.parameters(), target.parameters(), strict=False
        ):
            p_target.data.copy_(
                DDPG_TAU * p_online.data + (1 - DDPG_TAU) * p_target.data
            )

    def decay_noise(self) -> None:
        self.noise_sigma = max(
            DDPG_NOISE_SIGMA_END,
            self.noise_sigma * DDPG_NOISE_DECAY,
        )

    def save(self, path: str = "ddpg_agent.pth") -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )
        print(f"模型已儲存：{path}")

    def load(self, path: str = "ddpg_agent.pth") -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        print(f"模型已載入：{path}")
