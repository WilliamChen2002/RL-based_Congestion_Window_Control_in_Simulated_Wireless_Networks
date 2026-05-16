"""
agent.py — DQN Agent

約定：
  - Action : 0=減少 / 1=不變 / 2=增加 cwnd
  - State  : [throughput, rtt, loss_rate]（正規化，共 3 維）
  - Reward : 由環境計算後回傳（不在 agent 這邊算）
  - MAX_THROUGHPUT = 20 Mbps（router.bandwidth 上限）
  - MAX_RTT        = 75 ms  （base_delay=50 + max queue_delay=25）
  - 正規化由 env_wrapper.py 處理，agent 收到的 state 已是 0.0~1.0
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── 超參數（調參從這裡動） ──
STATE_DIM = 3  # [throughput, rtt, loss_rate]
ACTION_DIM = 3  # 0=減少 / 1=不變 / 2=增加

LR = 1e-3  # learning rate
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 64
BUFFER_SIZE = 10_000  # replay buffer 大小
LEARNING_STARTS = 500  # 幾步後才開始訓練
TARGET_UPDATE_FREQ = 500  # target network 更新頻率（步數）

EPS_START = 1.0  # 初始探索率
EPS_END = 0.05  # 最終探索率
EPS_DECAY = 0.99  # 每個 episode 乘上這個


# ── Q Network ──
class QNetwork(nn.Module):
    """
    3 層 MLP：state → Q value（每個 action 一個）
    輸入：正規化後的 state [throughput, rtt, loss_rate]
    輸出：[Q(s,減少), Q(s,不變), Q(s,增加)]
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
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


# ── Replay Buffer ──
class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ── DQN Agent ──
class DQNAgent:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)

        # Online network（主要訓練的）
        self.q_net = QNetwork().to(self.device)
        # Target network（穩定訓練用，定期從 q_net 複製）
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer()

        self.eps = EPS_START
        self._total_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        """ε-greedy：探索或利用"""
        if random.random() < self.eps:
            return random.randint(0, ACTION_DIM - 1)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax().item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """從 buffer 抽一批資料更新 Q network，回傳 loss"""
        if len(self.buffer) < LEARNING_STARTS:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).to(self.device)

        # 目前 Q 值
        q_values = self.q_net(states_t).gather(1, actions_t).squeeze(1)

        # Target Q 值（Bellman equation）
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1).values
            target = rewards_t + GAMMA * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._total_steps += 1

        # 定期更新 target network
        if self._total_steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self) -> None:
        """每個 episode 結束後呼叫，降低探索率"""
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    def save(self, path: str = "dqn_agent.pth") -> None:
        torch.save(self.q_net.state_dict(), path)
        print(f"模型已儲存：{path}")

    def load(self, path: str = "dqn_agent.pth") -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"模型已載入：{path}")
