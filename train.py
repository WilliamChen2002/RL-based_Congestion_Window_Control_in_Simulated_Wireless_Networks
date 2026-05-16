"""
train.py — 訓練主程式

用法：
  python train.py

真實環境 API 完成後，把 mock_env 換成真實環境即可，train.py 不用動。
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from agent import DQNAgent
from environment.env_wrapper import TCPEnvWrapper as MockEnv  # 真實環境 Wrapper

# ── 訓練設定 ──
N_EPISODES = 1000
EVAL_EVERY = 50  # 每幾個 episode 印一次進度


def train() -> tuple[DQNAgent, dict]:
    env = MockEnv()

    # 驗證環境符合 Gym 介面
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    agent = DQNAgent()

    history: dict[str, list] = {
        "reward": [],
        "throughput": [],
        "rtt": [],
        "loss": [],
        "eps": [],
    }

    for ep in range(1, N_EPISODES + 1):
        state, _ = env.reset(seed=42 if ep == 1 else None)
        done = False
        ep_reward = 0.0
        ep_info: dict[str, list] = {"throughput": [], "rtt": [], "loss": []}

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state
            ep_reward += reward
            ep_info["throughput"].append(info["throughput"])
            ep_info["rtt"].append(info["rtt"])
            ep_info["loss"].append(info["loss_rate"])

        agent.decay_epsilon()

        history["reward"].append(ep_reward)
        history["throughput"].append(float(np.mean(ep_info["throughput"])))
        history["rtt"].append(float(np.mean(ep_info["rtt"])))
        history["loss"].append(float(np.mean(ep_info["loss"])))
        history["eps"].append(agent.eps)

        if ep % EVAL_EVERY == 0:
            print(
                f"Episode {ep:4d} | "
                f"Reward {ep_reward:7.2f} | "
                f"Throughput {history['throughput'][-1]:5.1f} Mbps | "
                f"RTT {history['rtt'][-1]:5.1f} ms | "
                f"Loss {history['loss'][-1]*100:4.1f}% | "
                f"ε {agent.eps:.3f}"
            )

    agent.save()
    return agent, history


def plot(history: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("DQN Agent Training", fontsize=14)

    episodes = range(1, len(history["reward"]) + 1)

    def smooth(x: list, w: int = 10) -> np.ndarray:
        if len(x) < w:
            return np.array(x)
        return np.convolve(x, np.ones(w) / w, mode="valid")

    configs = [
        (axes[0, 0], history["reward"], "Episode Reward", "steelblue"),
        (axes[0, 1], history["throughput"], "Throughput (Mbps)", "darkorange"),
        (axes[1, 0], history["rtt"], "RTT (ms)", "crimson"),
        (axes[1, 1], history["loss"], "Packet Loss Rate", "purple"),
    ]

    for ax, data, title, color in configs:
        ax.plot(episodes, data, alpha=0.3, color=color)
        smoothed = smooth(data)
        ax.plot(range(1, len(smoothed) + 1), smoothed, color=color, lw=2)
        ax.set_title(title)
        ax.set_xlabel("Episode")

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"image/training_result_{timestamp}.png", dpi=150)
    print(f"圖表已儲存：image/training_result_{timestamp}.png")
    plt.show()


if __name__ == "__main__":
    agent, history = train()
    plot(history)
