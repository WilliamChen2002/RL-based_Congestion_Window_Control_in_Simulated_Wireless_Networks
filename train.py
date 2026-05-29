"""
train.py — 訓練模式選擇入口

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python train.py
"""

from train import train_dqn, plot_dqn
from train import train_ddpg, plot_ddpg

MODES = {
    "dqn":  (train_dqn,  plot_dqn),
    "ddpg": (train_ddpg, plot_ddpg),
}

if __name__ == "__main__":
    available = ", ".join(MODES.keys())
    print(f"可用的訓練模式：{available}")

    while True:
        mode = input("請輸入模式名稱：").strip().lower()
        if mode in MODES:
            break
        print(f"未知模式「{mode}」，請重新輸入。")

    train_fn, plot_fn = MODES[mode]
    agent, history = train_fn()
    plot_fn(history)
