import numpy as np

from environment import Receiver, Router, Sender


class TCP:
    def __init__(self):

        self.sender = Sender()

        self.router = Router()

        self.receiver = Receiver()

        self.max_steps = 30

        self.step_count = 0

    def reset(self):

        self.__init__()

        return self.get_state()

    def get_state(self):

        return np.array(
            [self.sender.cwnd, len(self.router.queue), self.router.wireless_loss],
            dtype=np.float32,
        )

    def step(self, action):

        # RL controls sender
        self.sender.apply_action(action)

        # sender generates packets
        packets = self.sender.send_packets()

        # router transmission
        delivered, dropped, rtt = self.router.transmit(packets)

        # receiver ACK
        ack_count = self.receiver.receive(delivered)

        throughput = ack_count

        loss_rate = dropped / max(len(packets), 1)

        reward = throughput - 0.1 * rtt - 20 * loss_rate

        self.step_count += 1

        done = self.step_count >= self.max_steps

        info = {
            "throughput": throughput,
            "loss_rate": loss_rate,
            "rtt": rtt,
            "queue": len(self.router.queue),
        }

        return (self.get_state(), reward, done, info)
