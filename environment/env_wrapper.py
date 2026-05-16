"""
env_wrapper.py — 真實環境 Wrapper

將 environment.TCP 包裝成符合 gymnasium.Env 的介面。
environment/ 資料夾需放在同一層目錄下。

正規化上限（依真實環境設定）：
  MAX_THROUGHPUT = 20   # router.bandwidth 上限
  MAX_RTT        = 75   # base_delay(50) + queue最大延遲(50*0.5=25)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .tcp import TCP

# ── 正規化上限（需與 agent.py 一致） ──
MAX_THROUGHPUT = 20.0  # ack_count 上限 = router.bandwidth
MAX_RTT = 75.0  # base_delay + max queue_delay


class TCPEnvWrapper(gym.Env):
    """
    把 environment.TCP 包成 Gym 介面。
    TCP 內部邏輯完全不動，只在外層轉換格式。
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()

        self._env = TCP()

        # Action：0=減少 / 1=不變 / 2=增加 cwnd
        self.action_space = spaces.Discrete(3)

        # Observation：[throughput, rtt, loss_rate]，正規化後 0.0~1.0
        self.observation_space = spaces.Box(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._env.reset()
        # reset 後還沒有 throughput/rtt/loss，給初始值
        obs = np.zeros(3, dtype=np.float32)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 呼叫原始環境（回傳 4 個值）
        _, reward, done, info = self._env.step(action)

        # 轉換 state：取 info 裡的原始值正規化
        obs = self._make_obs(info)

        # Gym 規範：done 拆成 terminated / truncated
        terminated = False
        truncated = done  # TCP 是時間到結束，屬於 truncated

        return obs, reward, terminated, truncated, info

    def _make_obs(self, info: dict) -> np.ndarray:
        throughput_norm = float(info["throughput"]) / MAX_THROUGHPUT
        rtt_norm = float(info["rtt"]) / MAX_RTT
        loss = float(info["loss_rate"])
        return np.clip(
            np.array([throughput_norm, rtt_norm, loss], dtype=np.float32),
            0.0,
            1.0,
        )
