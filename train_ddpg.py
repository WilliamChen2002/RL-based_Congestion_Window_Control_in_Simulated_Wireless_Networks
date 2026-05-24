"""
train_ddpg.py — DDPG 訓練主程式

用法：
  python train_ddpg.py

State  : [cwnd, throughput, aoi, loss_rate, queue]（正規化，5維）
Action : 目標 cwnd 值，範圍 [1.0, 100.0]（連續）
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from agent import DDPGAgent
from env_wrapper import DDPGEnvWrapper

# ── 訓練設定 ──
N_EPISODES = 1000   # 可調整
EVAL_EVERY = 50     # 每幾個 episode 印一次進度


def train() -> tuple[DDPGAgent, dict]:
    env = DDPGEnvWrapper()

    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)

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
        state, _ = env.reset(seed=42 if ep == 1 else None)
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
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            loss_info = agent.train_step()

            state = next_state
            ep_reward += reward
            ep_info["throughput"].append(info["network"]["throughput"])
            ep_info["aoi"].append(info["receiver"]["aoi"])
            ep_info["loss"].append(info["network"]["loss_rate"])
            ep_info["cwnd"].append(info["sender"]["cwnd"])

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
