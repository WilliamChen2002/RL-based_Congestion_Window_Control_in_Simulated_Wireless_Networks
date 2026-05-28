import numpy as np

from .agent_sender import AgentSender
from .cublic_sender import CubicSender
from .receiver import Receiver
from .reno_sender import RenoSender
from .router import Router


class TCPEnv:
    def __init__(self, mode="reno", seed=None):

        self.mode = mode
        self.rng = np.random.default_rng(seed)

        if mode == "reno":
            self.sender = RenoSender()
        elif mode == "cubic":
            self.sender = CubicSender()
        else:
            self.sender = AgentSender()

        self.router = Router(rng=self.rng)
        self.receiver = Receiver()

        self.max_steps = 50
        self.reset()

    def reset(self):
        self.time = 0
        self.step_count = 0

        self.sender.reset()
        self.router.reset()
        self.receiver.reset()

        return self.get_state(), {}

    def get_state(self):
        return np.array(
            [
                self.sender.cwnd,
                len(self.router.queue),
                self.router.base_bandwidth,
            ],
            dtype=np.float32,
        )

    def step(self, action=None, cwnd=None):

        self.time += 1
        self.step_count += 1

        if self.mode == "agent":
            self.sender.apply(action, cwnd)

        packets = self.sender.send(self.time)
        r = self.router.transmit(packets, self.time)

        ack, aoi = self.receiver.receive(r["delivered"], self.time)

        loss = r["congestion_drop"] + r["wireless_drop"]
        total_packets = max(1, len(packets))

        loss_rate = loss / total_packets
        rtt = 50 + r["queue_delay"] + r["transmit_time"]
        throughput = ack

        if loss > 0:
            self.sender.on_loss()
        else:
            self.sender.on_ack(ack)

        reward = throughput - 0.1 * rtt - 5 * loss_rate - 0.5 * aoi

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "network": {
                "throughput": throughput,
                "loss_rate": loss_rate,
                "rtt": rtt,
                "bandwidth": r["bandwidth"],
            },
            "router": {
                "queue_size": len(self.router.queue),
                "queue_delay": r["queue_delay"],
                "congestion_drop": r["congestion_drop"],
                "wireless_drop": r["wireless_drop"],
            },
            "sender": {
                "cwnd": self.sender.cwnd,
                "mode": self.mode,
            },
            "receiver": {
                "ack": ack,
                "aoi": aoi,
            },
        }

        return self.get_state(), reward, terminated, truncated, info
