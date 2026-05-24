"""
env_wrapper.py — 包裝 TCPEnv 成 gymnasium.Env 介面（訓練用）

功能：
  - 將 TCPEnv 的 step() 轉成 Gym 規範
  - reset() 補上 info dict
  - 正規化 state 成 [0.0, 1.0]
  - DQNEnvWrapper：離散 action {0, 1, 2}
  - DDPGEnvWrapper：連續 action，直接設定目標 cwnd 值

State（5維，正規化後各值 0.0～1.0）：
  [cwnd, throughput, aoi, loss_rate, queue]

正規化上限：
  MAX_CWND       = 100.0
  MAX_THROUGHPUT = 80.0
  MAX_AOI        = 30.0
  loss_rate      → 本來就是 0.0~1.0
  MAX_QUEUE      = 300.0

Reset 初始 state：
  cwnd=10（BaseSender 預設值），其他全 0
  正規化後：[0.1, 0.0, 0.0, 0.0, 0.0]

info 結構（env.py 巢狀）：
  info["sender"]["cwnd"]
  info["network"]["throughput"]
  info["network"]["loss_rate"]
  info["receiver"]["aoi"]
  info["router"]["queue_size"]
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import TCPEnv

# ── 正規化上限 ──
MAX_CWND = 100.0
MAX_THROUGHPUT = 80.0
MAX_AOI = 30.0
MAX_QUEUE = 300.0

# ── DDPG Action 範圍 ──
ACTION_LOW = 1.0
ACTION_HIGH = 100.0

# ── Episode 長度 ──
MAX_STEPS = 200


class DQNEnvWrapper(gym.Env):
    """
    給 DQN 用的 Wrapper。
    Action：離散 {0=減少, 1=不變, 2=增加}，由 AgentSender.apply(action=...) 處理。
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._env = TCPEnv(mode="agent")

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._env = TCPEnv(mode="agent")
        self._step_count = 0

        init_state = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return init_state, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"invalid action: {action}"

        _, reward, terminated, truncated, info = self._env.step(action=action)

        self._step_count += 1
        truncated = truncated or self._step_count >= MAX_STEPS

        obs = self._normalize(info)
        return obs, reward, terminated, truncated, info

    def _normalize(self, info: dict) -> np.ndarray:
        return np.array(
            [
                info["sender"]["cwnd"] / MAX_CWND,
                info["network"]["throughput"] / MAX_THROUGHPUT,
                info["receiver"]["aoi"] / MAX_AOI,
                info["network"]["loss_rate"],
                info["router"]["queue_size"] / MAX_QUEUE,
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)


class DDPGEnvWrapper(gym.Env):
    """
    給 DDPG 用的 Wrapper。
    Action：連續 [1.0, 100.0]，直接設定目標 cwnd。
    env.step() 只接 action，所以 DDPG 改成先設定 sender.cwnd 再呼叫 step(action=None)。
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self._env = TCPEnv(mode="agent")

        self.action_space = spaces.Box(
            low=np.array([ACTION_LOW], dtype=np.float32),
            high=np.array([ACTION_HIGH], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._env = TCPEnv(mode="agent")
        self._step_count = 0

        init_state = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return init_state, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # action 是 shape=(1,) 的 ndarray，取出 cwnd 值
        cwnd = float(np.clip(action[0], ACTION_LOW, ACTION_HIGH))

        # env.step() 不支援 cwnd 參數，直接設定 sender.cwnd 再呼叫 step
        self._env.sender.cwnd = cwnd
        _, reward, terminated, truncated, info = self._env.step(action=None)

        self._step_count += 1
        truncated = truncated or self._step_count >= MAX_STEPS

        obs = self._normalize(info)
        return obs, reward, terminated, truncated, info

    def _normalize(self, info: dict) -> np.ndarray:
        return np.array(
            [
                info["sender"]["cwnd"] / MAX_CWND,
                info["network"]["throughput"] / MAX_THROUGHPUT,
                info["receiver"]["aoi"] / MAX_AOI,
                info["network"]["loss_rate"],
                info["router"]["queue_size"] / MAX_QUEUE,
            ],
            dtype=np.float32,
        ).clip(0.0, 1.0)
