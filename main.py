import random

from tcp_env import WirelessTCPEnv

env = WirelessTCPEnv()

state = env.reset()

print("=== TCP RL Environment Start ===\n")

done = False

while not done:
    # random action: 0,1,2
    action = random.choice([0, 1, 2])

    next_state, reward, done, info = env.step(action)

    action_name = {0: "DECREASE", 1: "HOLD", 2: "INCREASE"}[action]

    print(f"Step: {env.step_count}")
    print(f"Action: {action_name}")

    print(f"CWND: {env.cwnd:.2f}")
    print(f"RTT: {env.rtt:.2f} ms")

    print(f"Loss: {info['loss']:.4f}")

    print(f"Throughput: {info['throughput']:.2f}")

    print(f"Reward: {reward:.2f}")

    print("-" * 40)

print("\n=== Simulation Finished ===")
