import random

import matplotlib.pyplot as plt

from environment import TCP

# 引入環境
env = TCP()

state = env.reset()

done = False

# 繪圖用串列紀錄值
steps = []
throughputs = []

# 隨機行動
while not done:
    # 行動設定(Agent該決定的事情)
    # 目前設定0 -> cwnd *= 0.7, 1 -> cwnd = cwnd, 2 -> cwnd *= 1.2
    action = random.choice([0, 1, 2])
    # next_state 路由器(bottleneck)的下一步狀態
    # reward 獎勵函數
    # done 目前設定模擬上限30步
    # info 顯示Throughput, Loss Rate, RTT, Queue(Router's)
    next_state, reward, done, info = env.step(action)

    # 印出來看成果
    print(f"Step {env.step_count}")
    print(f"CWND: {env.sender.cwnd:.2f}")
    print(f"Queue: {info['queue']}")
    print(f"RTT: {info['rtt']:.2f}")
    print(f"Loss: {info['loss_rate']:.2f}")
    print(f"Throughput: {info['throughput']}")
    print(f"Reward: {reward:.2f}")
    print("-" * 40)

    # 記錄到繪圖中
    steps.append(env.step_count)
    throughputs.append(info["throughput"])

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(steps, throughputs, marker="o", linestyle="-", color="b")
plt.title("Throughput over Steps")
plt.xlabel("Step")
plt.ylabel("Throughput")
plt.grid(True)
plt.show()
