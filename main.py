import random

import matplotlib.pyplot as plt

from environment import TCP

env = TCP()

state = env.reset()

done = False

steps = []
throughputs = []

while not done:
    action = random.choice([0, 1, 2])

    next_state, reward, done, info = env.step(action)

    print(f"Step {env.step_count}")
    print(f"CWND: {env.sender.cwnd:.2f}")
    print(f"Queue: {info['queue']}")
    print(f"RTT: {info['rtt']:.2f}")
    print(f"Loss: {info['loss_rate']:.2f}")
    print(f"Throughput: {info['throughput']}")
    print(f"Reward: {reward:.2f}")
    print("-" * 40)

    steps.append(env.step_count)
    throughputs.append(info["throughput"])

plt.figure(figsize=(10, 6))
plt.plot(steps, throughputs, marker="o", linestyle="-", color="b")
plt.title("Throughput over Steps")
plt.xlabel("Step")
plt.ylabel("Throughput")
plt.grid(True)
plt.show()
