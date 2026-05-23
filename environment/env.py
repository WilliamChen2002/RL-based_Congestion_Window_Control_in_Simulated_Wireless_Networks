import numpy as np
from .receiver import Receiver
from .router import Router
from .sender import BaseSender
from .reno_sender import RenoSender
from .agent_sender import AgentSender
from .cublic_sender import CubicSender


class TCPEnv:
    def __init__(self, sender_type="reno", max_steps=50):

        # sender selection
        if sender_type == "reno":
            self.sender = RenoSender()

        elif sender_type == "cubic":
            self.sender = CubicSender()

        elif sender_type == "rl":
            self.sender = AgentSender()

        else:
            raise ValueError("Unknown sender type")

        self.router = Router()

        self.receiver = Receiver()

        self.max_steps = max_steps

        self.current_time = 0

        self.step_count = 0

    # ========================================================
    # reset
    # ========================================================

    def reset(self):

        sender_type = type(self.sender).__name__

        if sender_type == "RenoSender":
            self.sender = RenoSender()

        elif sender_type == "CubicSender":
            self.sender = CubicSender()

        elif sender_type == "AgentSender":
            self.sender = AgentSender()

        self.router = Router()

        self.receiver = Receiver()

        self.current_time = 0

        self.step_count = 0

        return self.get_state()

    # ========================================================
    # state
    # ========================================================

    def get_state(self):

        return np.array(
            [self.sender.cwnd, len(self.router.queue), self.router.wireless_loss],
            dtype=np.float32,
        )

    # ========================================================
    # step
    # ========================================================

    def step(self, action=None):

        self.current_time += 1

        self.step_count += 1

        # RL action
        if isinstance(self.sender, AgentSender):
            self.sender.apply_action(action)

        # sender sends packets
        packets = self.sender.generate_packets(current_time=self.current_time)

        # router transmission
        router_result = self.router.transmit(packets)

        delivered = router_result["delivered"]

        # receiver ACK + AoI
        receiver_result = self.receiver.receive(delivered, self.current_time)

        ack_count = receiver_result["ack_count"]

        aoi = receiver_result["aoi"]

        # total loss
        total_drop = router_result["congestion_drop"] + router_result["wireless_drop"]

        # sender feedback
        if total_drop > 0:
            self.sender.on_loss()

        else:
            self.sender.on_ack(ack_count)

        # metrics
        throughput = ack_count

        sent_packets = max(1, len(packets))

        loss_rate = total_drop / sent_packets

        rtt = router_result["rtt"]

        queue_size = len(self.router.queue)

        # reward
        reward = throughput - 0.1 * rtt - 20 * loss_rate - 0.5 * aoi

        done = self.step_count >= self.max_steps

        info = {
            "throughput": throughput,
            "loss_rate": loss_rate,
            "rtt": rtt,
            "queue_size": queue_size,
            "aoi": aoi,
            "cwnd": self.sender.cwnd,
            "wireless_drop": router_result["wireless_drop"],
            "congestion_drop": router_result["congestion_drop"],
        }

        return (self.get_state(), reward, done, info)
