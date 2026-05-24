# client.py
import socket
import json
import matplotlib.pyplot as plt
import random
from datetime import datetime
import os


class TCPEnvClient:
    def __init__(self, host="127.0.0.1", port=8888):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def _send(self, data):
        msg = json.dumps(data) + "\n"
        self.sock.sendall(msg.encode())

    def _recv(self):
        data = b""
        while True:
            chunk = self.sock.recv(4096)
            data += chunk
            if b"\n" in data:
                break
        return json.loads(data.decode().strip())

    def create_session(self, mode: str, seed: int):
        self._send({"command": "create", "mode": mode, "seed": seed})
        return self._recv()

    def reset(self, session_id):
        self._send({"command": "reset", "session_id": session_id})
        return self._recv()

    def step(self, session_id, action=None):
        self._send({"command": "step", "session_id": session_id, "action": action})
        return self._recv()

    def close(self):
        self.sock.close()


def compare_modes(steps=150):
    # ==================== 建立記錄檔案 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)

    log_files = {}
    modes = ["reno", "cubic", "agent"]

    for mode in modes:
        filename = f"logs/{timestamp}_{mode.upper()}.txt"
        log_files[mode] = open(filename, "w", encoding="utf-8")
        log_files[mode].write(f"=== {mode.upper()} Simulation Log ===\n")
        log_files[mode].write(f"Timestamp : {datetime.now()}\n")
        log_files[mode].write(
            f"Random Seed: {random.randint(1, 100000)}\n"
        )  # 這裡會在下面覆蓋正確 seed
        log_files[mode].write("=" * 70 + "\n\n")

    # ==================== 主要模擬流程 ====================
    client = TCPEnvClient()
    base_seed = random.randint(1, 100000)
    print(f"使用相同隨機種子: {base_seed}\n")

    sessions = {}
    results = {mode: {"throughput": [], "aoi": []} for mode in modes}

    # 更新 log 檔頭的 seed
    for mode in modes:
        log_files[mode].write(f"Base Seed: {base_seed}\n\n")
        log_files[mode].write(
            f"Step    CWND    Throughput    AoI     Loss Rate    Queue\n"
        )
        log_files[mode].write("-" * 70 + "\n")

    # 建立三個獨立環境
    for mode in modes:
        resp = client.create_session(mode, base_seed)
        sessions[mode] = resp["session_id"]
        client.reset(sessions[mode])
        print(f"✅ 已建立 {mode.upper():6} 環境 → logs/{timestamp}_{mode.upper()}.txt")

    print("\n開始模擬...\n")

    for i in range(steps):
        for mode in modes:
            action = 1 if (mode == "agent" and i % 5 < 3) else None
            result = client.step(sessions[mode], action=action)
            info = result.get("info", {})

            # 儲存到結果（用來畫圖）
            results[mode]["throughput"].append(info.get("throughput", 0))
            results[mode]["aoi"].append(info.get("aoi", 0))

            # 寫入文字檔
            log_line = (
                f"{i + 1:4d}    {info.get('cwnd', 0):6.2f}    "
                f"{info.get('throughput', 0):8.2f}    "
                f"{info.get('aoi', 0):7.2f}    "
                f"{info.get('loss_rate', 0):8.3f}    "
                f"{info.get('queue', 0):5d}\n"
            )
            log_files[mode].write(log_line)

        # 每 30 step 顯示在畫面
        if i % 30 == 0:
            tp = {m: results[m]["throughput"][-1] for m in modes}
            print(
                f"Step {i + 1:3d} | Reno: {tp['reno']:.2f} | Cubic: {tp['cubic']:.2f} | Agent: {tp['agent']:.2f}"
            )

    # 關閉所有 log 檔案
    for f in log_files.values():
        f.close()

    client.close()

    print(f"\n✅ 模擬完成！三個詳細記錄已儲存至 logs/ 資料夾")
    print(f"檔案名稱：{timestamp}_(RENO|CUBIC|AGENT).txt\n")

    # ==================== 繪製比較圖 ====================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = range(steps)

    colors = {"reno": "blue", "cubic": "green", "agent": "red"}

    for mode in modes:
        ax1.plot(
            x,
            results[mode]["throughput"],
            label=mode.upper(),
            color=colors[mode],
            linewidth=2.2,
        )
        ax2.plot(
            x,
            results[mode]["aoi"],
            label=mode.upper(),
            color=colors[mode],
            linewidth=2.2,
        )

    ax1.set_title("Throughput Comparison (Independent TCPEnv)", fontsize=14)
    ax2.set_title("AoI Comparison (Independent TCPEnv)", fontsize=14)
    ax1.set_ylabel("Throughput")
    ax2.set_ylabel("AoI")
    ax2.set_xlabel("Step")
    ax1.legend()
    ax2.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Independent_TCPEnv_Comparison.png", dpi=300)
    print("✅ 比較圖已儲存: Independent_TCPEnv_Comparison.png")
    plt.show()


if __name__ == "__main__":
    compare_modes(steps=150)
