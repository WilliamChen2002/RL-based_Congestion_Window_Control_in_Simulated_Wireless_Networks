"""
compare.py — 四模式比較展示（需先啟動 env_server.py）

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python compare.py

比較模式：DQN / DDPG / Reno / Cubic
輸出：
  - logs/{timestamp}_{MODE}.txt
  - image/compare_{timestamp}.png

注意：
  - env_server.py 的 normalize_info() 會把巢狀 info 攤平成單層
  - 所以 compare.py 裡的 info 用單層 key：
    info["cwnd"], info["throughput"], info["aoi"], info["loss_rate"], info["queue"]
  - client.py 的 create_session() 已改名為 create()
"""

import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from agent import DDPGAgent, DQNAgent
from client import TCPEnvClient

# ── 設定 ──
STEPS = 150             # 可調整步數
DQN_MODEL_PATH = "dqn_agent.pth"
DDPG_MODEL_PATH = "ddpg_agent.pth"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8888

# ── 正規化上限（對齊 env_wrapper.py）──
MAX_CWND = 100.0
MAX_THROUGHPUT = 80.0
MAX_AOI = 30.0
MAX_QUEUE = 300.0


def normalize_state(info: dict) -> np.ndarray:
    # env_server 回傳的 info 是攤平的單層
    return np.array(
        [
            info.get("cwnd", 10.0) / MAX_CWND,
            info.get("throughput", 0.0) / MAX_THROUGHPUT,
            info.get("aoi", 0.0) / MAX_AOI,
            info.get("loss_rate", 0.0),
            info.get("queue", 0) / MAX_QUEUE,
        ],
        dtype=np.float32,
    ).clip(0.0, 1.0)


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
    # env_server 回傳的 info 是攤平的單層
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
    modes = ["dqn", "ddpg", "reno", "cubic"]

    print(f"Base Seed: {base_seed}")
    print(f"Steps    : {steps}\n")

    dqn_agent = DQNAgent()
    dqn_agent.load(DQN_MODEL_PATH)

    ddpg_agent = DDPGAgent()
    ddpg_agent.load(DDPG_MODEL_PATH)

    log_files = init_log_files(timestamp, modes, base_seed)

    client = TCPEnvClient(host=SERVER_HOST, port=SERVER_PORT)
    sessions = {}
    states = {}

    for mode in modes:
        env_mode = mode if mode in ("reno", "cubic") else "agent"
        # client.py 的方法已改名為 create()
        resp = client.create(mode=env_mode, seed=base_seed)
        sessions[mode] = resp["session_id"]
        client.reset(sessions[mode])
        states[mode] = np.array([0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        print(f"✅ 已建立 {mode.upper():6} 環境")

    print("\n開始模擬...\n")

    results = {
        mode: {"throughput": [], "aoi": [], "loss_rate": [], "cwnd": []}
        for mode in modes
    }

    for i in range(steps):
        for mode in modes:
            if mode == "dqn":
                action = dqn_agent.select_action(states[mode])
                _, _, _, info = client.step(sessions[mode], action=int(action))

            elif mode == "ddpg":
                action = ddpg_agent.select_action(states[mode], explore=False)
                cwnd_val = float(action[0])
                # DDPG 直接傳 cwnd 給 env_server
                client._send(
                    {
                        "command": "step",
                        "session_id": sessions[mode],
                        "action": None,
                        "cwnd": cwnd_val,
                    }
                )
                resp = client._recv()
                info = resp.get("info", {})

            else:
                _, _, _, info = client.step(sessions[mode], action=None)

            states[mode] = normalize_state(info)

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
                f"DDPG: {tp['ddpg']:5.2f} | "
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
        "dqn": "red",
        "ddpg": "darkorange",
        "reno": "blue",
        "cubic": "green",
    }
    labels = {
        "dqn": "DQN",
        "ddpg": "DDPG",
        "reno": "Reno",
        "cubic": "Cubic",
    }
    plot_modes = ["dqn", "ddpg", "reno", "cubic"]

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
                linewidth=2.0,
                alpha=0.85,
            )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"image/compare_{timestamp}.png"
    plt.savefig(path, dpi=150)
    print(f"✅ 比較圖已儲存：{path}")
    plt.show()


if __name__ == "__main__":
    compare(steps=STEPS)
