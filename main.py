# import random

from environment import TCPEnv


def run_simulation(mode, type=None, action=None, cwnd=None, coef=None):

    print(f"\n===== {mode.upper()} =====\n")

    env = TCPEnv(mode=mode)

    state = env.reset()

    done = False

    while not done:
        action = None
        # Agent更改部分
        if mode == "agent":
            if type == "DQN":
                # Agent選擇的部分(DQN)
                action = action
                coef = coef
                # action = random.choice([0, 1, 2])
            if type == "DDPG":
                # DDPG
                action = action
                cwnd = cwnd
                # action = random.randint(1, 20)

        next_state, reward, done, info = env.step(action=action, cwnd=cwnd, coef=coef)

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

    # run_simulation("cubic")

    # 這行是給DQN用的 action 是你 agent 選擇的動作 action，動作跟上一版相同等待討論5/23
    # run_simulation("agent", "DQN", action=agent_action, cwnd=None)
    # 這行是給DDPG用的 cwnd 是你 agent 輸出的cwnd值
    # run_simulation("agent", "DDPG", action=None, cwnd=agent_cwnd)
    # 這行是給DDPG用的 coef 是你 agent 輸出的coef值
    # run_simulation("agent", "DDPG", action=None, cwnd=None, coef=coef)
    # run_simulation("agent", "DDPG", action=None, cwnd=agent_cwnd, coef=coef)

    run_simulation("agent", "DDPG", action=None, cwnd=None, coef=1.2)
