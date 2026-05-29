"""
train_dqn_no_aoi.py — DQN 訓練主程式（無 AoI 消融版）

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python train.py  → 選擇 dqn_no_aoi

State（4維，拿掉 aoi）：[cwnd, throughput, loss_rate, queue]
Action：0=減少 / 1=不變 / 2=增加 cwnd（離散）
Reward：throughput - 0.1 * rtt - 5 * loss_rate（aoi_request=None，環境不含 aoi 項）
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from client import TCPEnvClient
from train.agent import DQNAgent

# ── 訓練設定 ──
N_EPISODES = 1000
EVAL_EVERY  = 50
SEED        = 42

# ── 正規化上限 ──
MAX_CWND       = 100.0
MAX_THROUGHPUT = 80.0
MAX_QUEUE      = 300.0

# DQN 輸入維度（無 AoI）
STATE_DIM_NO_AOI = 4


def make_state(info: dict) -> np.ndarray:
    """把 info 組成正規化 4 維 state（不含 aoi）。"""
    return np.array(
        [
            info["cwnd"]       / MAX_CWND,
            info["throughput"] / MAX_THROUGHPUT,
            info["loss_rate"],
            info["queue"]      / MAX_QUEUE,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)


def make_init_state() -> np.ndarray:
    """reset 時尚無 info，用初始值（4維）。"""
    return np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)


def train_dqn_no_aoi() -> tuple[DQNAgent, dict]:
    client = TCPEnvClient()
    agent  = DQNAgent(state_dim=STATE_DIM_NO_AOI)

    history: dict[str, list] = {
        "reward":     [],
        "throughput": [],
        "aoi":        [],
        "loss":       [],
        "eps":        [],
        "q_loss":     [],
    }

    for ep in range(1, N_EPISODES + 1):
        resp       = client.create(mode="agent", seed=SEED if ep == 1 else ep)
        session_id = resp["session_id"]
        client.reset(session_id)

        state     = make_init_state()
        done      = False
        ep_reward = 0.0
        ep_info: dict[str, list] = {
            "throughput": [],
            "aoi":        [],
            "loss":       [],
            "q_loss":     [],
        }

        while not done:
            action = agent.select_action(state)

            _, reward, done, info = client.step(
                session_id,
                action=action,
                cwnd=None,
                # aoi_request 不傳 → 預設 None → reward 不含 aoi
            )

            next_state = make_state(info)

            agent.store(state, action, reward, next_state, done)
            q_loss = agent.train_step()

            state      = next_state
            ep_reward += reward
            ep_info["throughput"].append(info["throughput"])
            ep_info["aoi"].append(info["aoi"])  # 仍記錄 aoi 供觀察
            ep_info["loss"].append(info["loss_rate"])

            if q_loss is not None:
                ep_info["q_loss"].append(q_loss)

        agent.decay_epsilon()

        history["reward"].append(ep_reward)
        history["throughput"].append(float(np.mean(ep_info["throughput"])))
        history["aoi"].append(float(np.mean(ep_info["aoi"])))
        history["loss"].append(float(np.mean(ep_info["loss"])))
        history["eps"].append(agent.eps)
        history["q_loss"].append(
            float(np.mean(ep_info["q_loss"])) if ep_info["q_loss"] else 0.0
        )

        if ep % EVAL_EVERY == 0:
            print(
                f"[DQN-noAoI] Episode {ep:4d} | "
                f"Reward {ep_reward:8.2f} | "
                f"Throughput {history['throughput'][-1]:5.1f} | "
                f"AoI {history['aoi'][-1]:5.2f} | "
                f"Loss {history['loss'][-1]*100:4.1f}% | "
                f"ε {agent.eps:.3f} | "
                f"Q_loss {history['q_loss'][-1]:.4f}"
            )

    client.close()
    os.makedirs("model", exist_ok=True)
    agent.save("model/dqn_no_aoi.pth")
    return agent, history


def plot_dqn_no_aoi(history: dict) -> None:
    os.makedirs("figures", exist_ok=True)

    episodes = range(1, len(history["reward"]) + 1)

    configs = [
        ("reward",     "Episode Reward",    "steelblue"),
        ("throughput", "Throughput (pkts)", "darkorange"),
        ("q_loss",     "Q Loss",            "tomato"),
        ("aoi",        "AoI",               "crimson"),
        ("loss",       "Packet Loss Rate",  "purple"),
        ("eps",        "Epsilon",           "gray"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("DQN Agent Training (No AoI)", fontsize=14)

    for ax, (key, title, color) in zip(axes.flatten(), configs):
        ax.plot(episodes, history[key], color=color)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"figures/dqn_no_aoi_training_{timestamp}.png"
    plt.savefig(path, dpi=150)
    print(f"圖表已儲存：{path}")
    plt.show()
