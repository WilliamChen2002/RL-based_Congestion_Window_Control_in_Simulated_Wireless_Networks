import random

from environment import TCPEnv


def run_simulation(mode):

    print(f"\n===== {mode.upper()} =====\n")

    env = TCPEnv(mode=mode)

    state = env.reset()

    done = False

    while not done:
        # RL only
        action = None

        if mode == "agent":
            action = random.choice([0, 1, 2])

        next_state, reward, done, info = env.step(action)

        state = next_state

        print(f"Step: {env.step_count}")

        print(f"CWND: {info['cwnd']:.2f}")

        print(f"Throughput: {info['throughput']}")

        print(f"RTT: {info['rtt']:.2f}")

        print(f"Loss Rate: {info['loss_rate']:.3f}")

        print(f"Queue Size: {info['queue_size']}")

        print(f"AoI: {info['aoi']}")

        print(f"Reward: {reward:.2f}")

        print(f"State {state}")

        print("-" * 40)


if __name__ == "__main__":
    # run_simulation("reno")

    run_simulation("cubic")

    # run_simulation("agent")
