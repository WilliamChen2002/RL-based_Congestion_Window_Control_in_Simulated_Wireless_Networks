"""
train_ddpg.py — DDPG 訓練主程式

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python train_ddpg.py

State（5維，從 info 組成後正規化）：
  [cwnd, throughput, aoi, loss_rate, queue]

Action：目標 cwnd 值，範圍 [1.0, 100.0]（連續）
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from train.agent import DDPGAgent
from client import TCPEnvClient

# ── 訓練設定 ──
N_EPISODES = 1000
EVAL_EVERY = 50
SEED = 42

# ── 正規化上限 ──
MAX_CWND = 100.0
MAX_THROUGHPUT = 80.0
MAX_AOI = 30.0
MAX_QUEUE = 300.0


def make_state(info: dict) -> np.ndarray:
    """把 info 組成正規化 5 維 state。"""
    return np.array(
        [
            info["cwnd"]       / MAX_CWND,
            info["throughput"] / MAX_THROUGHPUT,
            info["aoi"]        / MAX_AOI,
            info["loss_rate"],
            info["queue"]      / MAX_QUEUE,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)


def make_init_state() -> np.ndarray:
    """reset 時尚無 info，用初始值。"""
    return np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def train() -> tuple[DDPGAgent, dict]:
    client = TCPEnvClient()
    agent = DDPGAgent()

    history: dict[str, list] = {
        "reward": [],
        "throughput": [],
        "aoi": [],
        "loss": [],
        "cwnd": [],
        "noise": [],
        "actor_loss": [],
        "critic_loss": [],
    }

    for ep in range(1, N_EPISODES + 1):
        resp = client.create(mode="agent", seed=SEED if ep == 1 else ep)
        session_id = resp["session_id"]
        client.reset(session_id)

        state = make_init_state()
        done = False
        ep_reward = 0.0
        ep_info: dict[str, list] = {
            "throughput": [],
            "aoi": [],
            "loss": [],
            "cwnd": [],
            "actor_loss": [],
            "critic_loss": [],
        }

        while not done:
            action = agent.select_action(state, explore=True)
            cwnd_val = float(action[0])

            _, reward, done, info = client.step(
                session_id,
                action=None,
                cwnd=cwnd_val,
            )

            next_state = make_state(info)

            agent.store(state, action, reward, next_state, done)
            loss_info = agent.train_step()

            state = next_state
            ep_reward += reward
            ep_info["throughput"].append(info["throughput"])
            ep_info["aoi"].append(info["aoi"])
            ep_info["loss"].append(info["loss_rate"])
            ep_info["cwnd"].append(info["cwnd"])

            if loss_info is not None:
                ep_info["actor_loss"].append(loss_info["actor_loss"])
                ep_info["critic_loss"].append(loss_info["critic_loss"])

        agent.decay_noise()

        history["reward"].append(ep_reward)
        history["throughput"].append(float(np.mean(ep_info["throughput"])))
        history["aoi"].append(float(np.mean(ep_info["aoi"])))
        history["loss"].append(float(np.mean(ep_info["loss"])))
        history["cwnd"].append(float(np.mean(ep_info["cwnd"])))
        history["noise"].append(agent.noise_sigma)
        history["actor_loss"].append(
            float(np.mean(ep_info["actor_loss"])) if ep_info["actor_loss"] else 0.0
        )
        history["critic_loss"].append(
            float(np.mean(ep_info["critic_loss"])) if ep_info["critic_loss"] else 0.0
        )

        if ep % EVAL_EVERY == 0:
            a_loss = history["actor_loss"][-1]
            c_loss = history["critic_loss"][-1]
            print(
                f"[DDPG] Episode {ep:4d} | "
                f"Reward {ep_reward:8.2f} | "
                f"Throughput {history['throughput'][-1]:5.1f} | "
                f"AoI {history['aoi'][-1]:5.2f} | "
                f"Loss {history['loss'][-1]*100:4.1f}% | "
                f"CWND {history['cwnd'][-1]:6.1f} | "
                f"σ {agent.noise_sigma:.2f} | "
                f"A_loss {a_loss:.4f} | C_loss {c_loss:.4f}"
            )

    client.close()
    agent.save("ddpg_agent.pth")
    return agent, history


def plot(history: dict) -> None:
    import os
    os.makedirs("image", exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle("DDPG Agent Training", fontsize=14)

    episodes = range(1, len(history["reward"]) + 1)

    def smooth(x: list, w: int = 10) -> np.ndarray:
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    configs = [
        (axes[0, 0], history["reward"],      "Episode Reward",    "steelblue"),
        (axes[0, 1], history["throughput"],  "Throughput (pkts)", "darkorange"),
        (axes[0, 2], history["cwnd"],        "CWND",              "teal"),
        (axes[1, 0], history["aoi"],         "AoI",               "crimson"),
        (axes[1, 1], history["loss"],        "Packet Loss Rate",  "purple"),
        (axes[1, 2], history["noise"],       "Noise σ",           "gray"),
        (axes[2, 0], history["actor_loss"],  "Actor Loss",        "tomato"),
        (axes[2, 1], history["critic_loss"], "Critic Loss",       "royalblue"),
    ]

    for ax, data, title, color in configs:
        ax.plot(episodes, data, alpha=0.3, color=color)
        smoothed = smooth(data)
        ax.plot(range(1, len(smoothed) + 1), smoothed, color=color, lw=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")

    axes[2, 2].axis("off")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"image/ddpg_training_{timestamp}.png"
    plt.savefig(path, dpi=150)
    print(f"圖表已儲存：{path}")
    plt.show()


if __name__ == "__main__":
    agent, history = train()
    plot(history)
