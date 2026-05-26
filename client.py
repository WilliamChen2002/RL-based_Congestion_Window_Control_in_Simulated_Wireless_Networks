import json
import os
import random
import socket
from datetime import datetime

import matplotlib.pyplot as plt


# =========================================================
# CLEAN RL-STYLE TCP CLIENT (API VERSION)
# =========================================================
class TCPEnvClient:
    def __init__(self, host="127.0.0.1", port=8888):

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    # -----------------------------
    # internal IO
    # -----------------------------
    def _send(self, data):
        self.sock.sendall((json.dumps(data) + "\n").encode())

    def _recv(self):
        buffer = b""
        while True:
            chunk = self.sock.recv(4096)
            buffer += chunk
            if b"\n" in buffer:
                break
        return json.loads(buffer.decode().strip())

    # =====================================================
    # PUBLIC API (FOR AGENT DEVELOPERS)
    # =====================================================

    def create(self, mode="agent", seed=42):
        """
        Create environment session
        """
        self._send({"command": "create", "mode": mode, "seed": seed})
        return self._recv()

    def reset(self, session_id):
        """
        Reset environment -> returns initial state
        """
        self._send({"command": "reset", "session_id": session_id})

        res = self._recv()
        return res["state"]

    def step(self, session_id, action, cwnd):
        """
        RL standard interface:

        Returns:
            obs, reward, done, info
        """

        self._send(
            {
                "command": "step",
                "session_id": session_id,
                "action": action,
                "cwnd": cwnd,
            }
        )

        res = self._recv()

        return (res["state"], res["reward"], res.get("done", False), res["info"])

    def close(self):
        self.sock.close()


# =========================================================
# ACTION WRAPPER (FOR MULTI-AGENT SUPPORT)
# =========================================================
def build_action(agent_type, agent, state):

    if agent_type == "dqn":
        return agent.act(state)  # discrete: 0/1/2

    elif agent_type == "ddpg":
        return agent.act(state)  # continuous cwnd

    elif agent_type == "coef":
        return agent.act(state)  # scaling factor

    else:
        return 1


# =========================================================
# EPISODE RUNNER (CLEAN RL LOOP)
# =========================================================
def run_episode(client, session_id, agent, agent_type, steps=150):

    state = client.reset(session_id)

    trajectory = []

    for _t in range(steps):
        # ===== 1. agent action =====
        action = build_action(agent_type, agent, state)

        # ===== 2. env step =====
        next_state, reward, done, info = client.step(session_id, action)

        # ===== 3. store transition =====
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "info": info,
        }

        trajectory.append(transition)

        # ===== 4. learning hook =====
        if hasattr(agent, "store"):
            agent.store(transition)

        if hasattr(agent, "learn"):
            agent.learn()

        state = next_state

        if done:
            break

    return trajectory


# =========================================================
# MULTI-MODE EXPERIMENT (RENO / CUBIC / AGENT)
# =========================================================
def run_experiment(steps=150):

    client = TCPEnvClient()

    modes = ["reno", "cubic", "agent"]
    seed = 42

    sessions = {}

    results = {
        m: {
            "cwnd": [],
            "throughput": [],
            "aoi": [],
            "loss": [],
        }
        for m in modes
    }

    # -----------------------------
    # create sessions
    # -----------------------------
    for m in modes:
        resp = client.create(mode=m, seed=seed)
        sid = resp["session_id"]
        client.reset(sid)
        sessions[m] = sid

    print("🚀 simulation start...\n")

    # -----------------------------
    # main loop
    # -----------------------------
    for t in range(steps):
        for m in modes:
            sid = sessions[m]

            # ===== simple baseline agent =====
            if m == "agent":
                action, cwnd = agent_choice()
            else:
                action = None
                cwnd = None

            _, _, _, info = client.step(sid, action, cwnd)

            results[m]["cwnd"].append(info["cwnd"])
            results[m]["throughput"].append(info["throughput"])
            results[m]["aoi"].append(info["aoi"])
            results[m]["loss"].append(info["loss_rate"])
            agent_get_info(client.step(sid, action, cwnd))

        if t % 30 == 0:
            print(f"Step {t} done")

    client.close()
    return results


# =========================================================
# PLOTTING (4 METRICS + COMBINED)
# =========================================================
def plot_results(results, steps):

    x = range(steps)

    os.makedirs("figures", exist_ok=True)

    colors = {"reno": "blue", "cubic": "green", "agent": "red"}

    metrics = ["cwnd", "throughput", "aoi", "loss"]

    # =====================================================
    # single plots
    # =====================================================
    for metric in metrics:
        plt.figure(figsize=(10, 5))

        for m in results:
            plt.plot(x, results[m][metric], label=m.upper(), color=colors[m])

        plt.title(metric.upper())
        plt.xlabel("Step")
        plt.legend()
        plt.grid(alpha=0.3)

        path = f"figures/{metric}.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"📁 saved: {path}")

    # =====================================================
    # combined plot
    # =====================================================
    fig, axs = plt.subplots(4, 1, figsize=(14, 12))

    for m in results:
        axs[0].plot(x, results[m]["cwnd"], label=m.upper(), color=colors[m])
        axs[1].plot(x, results[m]["throughput"], label=m.upper(), color=colors[m])
        axs[2].plot(x, results[m]["aoi"], label=m.upper(), color=colors[m])
        axs[3].plot(x, results[m]["loss"], label=m.upper(), color=colors[m])

    axs[0].set_title("CWND")
    axs[1].set_title("Throughput")
    axs[2].set_title("AoI")
    axs[3].set_title("Loss Rate")

    for ax in axs:
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    path = f"figures/combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(path, dpi=300)

    print(f"📊 saved combined: {path}")

    plt.show()


# Give env agent choice(Agent control)
def agent_choice():
    action, cwnd = simulate_agent()
    return action, cwnd


# Env feedback
def agent_get_info(info):
    while True:
        read_ok = agent_read_status()
        if read_ok:
            break
    print(info)


# Simulate agent action: DQN set action 0, 1, 2; DDPG set cwnd
def simulate_agent():
    action = None
    cwnd = random.randint(10, 20)
    print(f"Agent do: actio={action}, cwnd={cwnd}")
    return action, cwnd


# Agent read dat time when agent read data over, return True else False or don't return
def agent_read_status():
    import time

    print("Agent reading data")
    time.sleep(1)
    return True


# =========================================================
# MAIN ENTRY
# =========================================================
if __name__ == "__main__":
    steps = 150

    results = run_experiment(steps)

    plot_results(results, steps)

    print("\n✅ finished")
