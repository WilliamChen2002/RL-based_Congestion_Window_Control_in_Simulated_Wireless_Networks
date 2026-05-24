import numpy as np

from .agent_sender import AgentSender
from .cublic_sender import CubicSender
from .receiver import Receiver
from .reno_sender import RenoSender
from .router import Router


class TCPEnv:
    def __init__(self, mode="reno"):

        if mode == "reno":
            self.sender = RenoSender()

        elif mode == "cubic":
            self.sender = CubicSender()

        else:
            self.sender = AgentSender()

        self.router = Router()

        self.receiver = Receiver()

        self.time = 0

        self.step_count = 0

        self.mode = mode

    def reset(self):

        return self.get_state()

    def get_state(self):

        return np.array(
            [self.sender.cwnd, len(self.router.queue), self.router.base_bandwidth],
            dtype=np.float32,
        )

    def step(self, action=None, cwnd=None, coef=None):

        self.time += 1

        self.step_count += 1

        # sender
        if self.mode == "agent":
            self.sender.apply(action=action, cwnd=cwnd, coef=coef)

        packets = self.sender.send(self.time)

        # router
        r = self.router.transmit(packets, self.time)

        # receiver
        ack, aoi = self.receiver.receive(r["delivered"], self.time)

        loss = r["congestion_drop"] + r["wireless_drop"]

        total_packets = max(1, len(packets))

        loss_rate = loss / total_packets

        rtt = 50 + r["queue_delay"] + r["transmit_time"]

        # sender update
        if loss > 0:
            self.sender.on_loss()

        else:
            self.sender.on_ack(ack)

        throughput = ack

        # ===== reward (KEY UPGRADE) =====
        reward = throughput - 0.1 * rtt - 20 * loss_rate - 0.5 * aoi

        done = self.step_count > 50

        info = {
            "throughput": throughput,
            "loss_rate": loss_rate,
            "rtt": rtt,
            "aoi": aoi,
            "queue": len(self.router.queue),
            "bandwidth": r["bandwidth"],
            "tx_time": r["transmit_time"],
            "cwnd": self.sender.cwnd,
            "queue_size": len(self.router.queue),
        }

        return self.get_state(), reward, done, info
