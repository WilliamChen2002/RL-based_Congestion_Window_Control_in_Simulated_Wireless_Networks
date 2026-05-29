"""
compare.py — 六模式比較展示（DQN / DQN-noAoI / DDPG / DDPG-noAoI / Reno / Cubic）

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python compare.py
"""

import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from train.agent import DDPGAgent, DQNAgent
from client import TCPEnvClient

# ── 設定 ──
STEPS            = 150
DQN_MODEL_PATH         = "model/dqn_agent.pth"
DDPG_MODEL_PATH        = "model/ddpg_agent.pth"
DQN_NO_AOI_MODEL_PATH  = "model/dqn_no_aoi.pth"
DDPG_NO_AOI_MODEL_PATH = "model/ddpg_no_aoi.pth"

# ── 正規化上限 ──
MAX_CWND       = 100.0
MAX_THROUGHPUT = 80.0
MAX_AOI        = 30.0
MAX_QUEUE      = 300.0


def make_state(info: dict) -> np.ndarray:
    """5 維 state（含 aoi）。"""
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


def make_state_no_aoi(info: dict) -> np.ndarray:
    """4 維 state（不含 aoi）。"""
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
    return np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_init_state_no_aoi() -> np.ndarray:
    return np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)


def init_log_files(timestamp: str, modes: list, base_seed: int) -> dict:
    os.makedirs("logs", exist_ok=True)
    log_files = {}

    for mode in modes:
        path = f"logs/{timestamp}_{mode.upper()}.txt"
        f = open(path, "w", encoding="utf-8")
        f.write(f"=== {mode.upper()} Simulation Log ===\n")
        f.write(f"Timestamp : {datetime.now()}\n")
        f.write(f"Base Seed : {base_seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Step    CWND    Throughput    AoI     Loss Rate    Queue\n")
        f.write("-" * 70 + "\n")
        log_files[mode] = f

    return log_files


def write_log(f, step: int, info: dict) -> None:
    f.write(
        f"{step:4d}    {info.get('cwnd', 0):6.2f}    "
        f"{info.get('throughput', 0):8.2f}    "
        f"{info.get('aoi', 0):7.2f}    "
        f"{info.get('loss_rate', 0):8.3f}    "
        f"{info.get('queue', 0):5d}\n"
    )


def compare(steps: int = STEPS) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_seed = random.randint(1, 100000)
    modes = ["dqn", "dqn_no_aoi", "ddpg", "ddpg_no_aoi", "reno", "cubic"]

    print(f"Base Seed: {base_seed}")
    print(f"Steps    : {steps}\n")

    # ── 載入模型 ──
    dqn_agent = DQNAgent(state_dim=5)
    dqn_agent.load(DQN_MODEL_PATH)

    dqn_no_aoi_agent = DQNAgent(state_dim=4)
    dqn_no_aoi_agent.load(DQN_NO_AOI_MODEL_PATH)

    ddpg_agent = DDPGAgent(state_dim=5)
    ddpg_agent.load(DDPG_MODEL_PATH)

    ddpg_no_aoi_agent = DDPGAgent(state_dim=4)
    ddpg_no_aoi_agent.load(DDPG_NO_AOI_MODEL_PATH)

    log_files = init_log_files(timestamp, modes, base_seed)

    client = TCPEnvClient()
    sessions = {}
    states   = {}

    for mode in modes:
        env_mode = mode if mode in ("reno", "cubic") else "agent"
        resp = client.create(mode=env_mode, seed=base_seed)
        sessions[mode] = resp["session_id"]
        client.reset(sessions[mode])

        if "no_aoi" in mode:
            states[mode] = make_init_state_no_aoi()
        else:
            states[mode] = make_init_state()

        print(f"✅ 已建立 {mode.upper():12} 環境")

    print("\n開始模擬...\n")

    results = {
        mode: {"throughput": [], "aoi": [], "loss_rate": [], "cwnd": []}
        for mode in modes
    }

    for i in range(steps):
        for mode in modes:
            if mode == "dqn":
                action = dqn_agent.select_action(states[mode])
                _, _, _, info = client.step(
                    sessions[mode],
                    action=int(action),
                    cwnd=None,
                    aoi_request=1,
                )

            elif mode == "dqn_no_aoi":
                action = dqn_no_aoi_agent.select_action(states[mode])
                _, _, _, info = client.step(
                    sessions[mode],
                    action=int(action),
                    cwnd=None,
                )

            elif mode == "ddpg":
                action   = ddpg_agent.select_action(states[mode], explore=False)
                cwnd_val = float(action[0])
                _, _, _, info = client.step(
                    sessions[mode],
                    action=None,
                    cwnd=cwnd_val,
                    aoi_request=1,
                )

            elif mode == "ddpg_no_aoi":
                action   = ddpg_no_aoi_agent.select_action(states[mode], explore=False)
                cwnd_val = float(action[0])
                _, _, _, info = client.step(
                    sessions[mode],
                    action=None,
                    cwnd=cwnd_val,
                )

            else:
                _, _, _, info = client.step(
                    sessions[mode],
                    action=None,
                    cwnd=None,
                )

            if "no_aoi" in mode:
                states[mode] = make_state_no_aoi(info)
            else:
                states[mode] = make_state(info)

            results[mode]["throughput"].append(info.get("throughput", 0))
            results[mode]["aoi"].append(info.get("aoi", 0))
            results[mode]["loss_rate"].append(info.get("loss_rate", 0))
            results[mode]["cwnd"].append(info.get("cwnd", 0))

            write_log(log_files[mode], i + 1, info)

        if (i + 1) % 30 == 0:
            tp = {m: results[m]["throughput"][-1] for m in modes}
            print(
                f"Step {i + 1:3d} | "
                f"DQN: {tp['dqn']:5.2f} | "
                f"DQN-noAoI: {tp['dqn_no_aoi']:5.2f} | "
                f"DDPG: {tp['ddpg']:5.2f} | "
                f"DDPG-noAoI: {tp['ddpg_no_aoi']:5.2f} | "
                f"Reno: {tp['reno']:5.2f} | "
                f"Cubic: {tp['cubic']:5.2f}"
            )

    for f in log_files.values():
        f.close()

    client.close()

    print(f"\n✅ 模擬完成！Log 已儲存至 logs/{timestamp}_*.txt\n")

    plot(results, steps, timestamp)


def plot(results: dict, steps: int, timestamp: str) -> None:
    os.makedirs("image", exist_ok=True)

    colors = {
        "dqn":          "#E57373",   # 紅
        "dqn_no_aoi":   "#81C784",   # 綠
        "ddpg":         "#FFB74D",   # 橙
        "ddpg_no_aoi":  "#64B5F6",   # 藍
        "reno":         "#BA68C8",   # 紫
        "cubic":        "#A1887F",   # 棕
    }
    labels = {
        "dqn":          "DQN",
        "dqn_no_aoi":   "DQN (No AoI)",
        "ddpg":         "DDPG",
        "ddpg_no_aoi":  "DDPG (No AoI)",
        "reno":         "Reno",
        "cubic":        "Cubic",
    }
    linestyles = {
        "dqn":          "-",
        "dqn_no_aoi":   "-",
        "ddpg":         "-",
        "ddpg_no_aoi":  "-",
        "reno":         "-",
        "cubic":        "-",
    }

    plot_modes = ["dqn", "dqn_no_aoi", "ddpg", "ddpg_no_aoi", "reno", "cubic"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("TCP Congestion Control Comparison", fontsize=15)
    x = range(1, steps + 1)

    metrics = [
        (axes[0, 0], "throughput", "Throughput (pkts)"),
        (axes[0, 1], "aoi",        "AoI"),
        (axes[1, 0], "loss_rate",  "Loss Rate"),
        (axes[1, 1], "cwnd",       "CWND"),
    ]

    for ax, key, title in metrics:
        for mode in plot_modes:
            ax.plot(
                x,
                results[mode][key],
                label=labels[mode],
                color=colors[mode],
                linestyle=linestyles[mode],
                linewidth=2.0,
                alpha=0.9,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if key == "aoi":
            ymax = ax.get_ylim()[1]
            ax.set_ylim(bottom=0, top=max(ymax, 1))

    plt.tight_layout()
    path = f"image/compare_{timestamp}.png"
    plt.savefig(path, dpi=150)
    print(f"✅ 比較圖已儲存：{path}")
    plt.show()


if __name__ == "__main__":
    compare(steps=STEPS)
