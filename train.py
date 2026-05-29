"""
train.py — 訓練模式選擇入口

用法：
  1. 先啟動 server：python env_server.py
  2. 再執行：       python train.py
"""

from train.train_dqn         import train_dqn,         plot_dqn
from train.train_ddpg        import train_ddpg,         plot_ddpg
from train.train_dqn_no_aoi  import train_dqn_no_aoi,  plot_dqn_no_aoi
from train.train_ddpg_no_aoi import train_ddpg_no_aoi, plot_ddpg_no_aoi

MODES = {
    "dqn":         (train_dqn,         plot_dqn),
    "ddpg":        (train_ddpg,        plot_ddpg),
    "dqn_no_aoi":  (train_dqn_no_aoi,  plot_dqn_no_aoi),
    "ddpg_no_aoi": (train_ddpg_no_aoi, plot_ddpg_no_aoi),
}

if __name__ == "__main__":
    available = ", ".join(MODES.keys())

    while True:
        print(f"\n可用的訓練模式：{available}")
        print("輸入 q 離開")

        mode = input("請輸入模式名稱：").strip().lower()

        if mode == "q":
            print("結束訓練。")
            break

        if mode not in MODES:
            print(f"未知模式「{mode}」，請重新輸入。")
            continue

        train_fn, plot_fn = MODES[mode]
        agent, history = train_fn()
        plot_fn(history)
