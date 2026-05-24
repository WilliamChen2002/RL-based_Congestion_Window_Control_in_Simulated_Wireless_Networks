from abc import ABC, abstractmethod

from .packet import Packet


class BaseSender(ABC):
    def __init__(self):

        self.init_cwnd = 10.0
        self.init_ssthresh = 32.0

        self.cwnd = self.init_cwnd
        self.ssthresh = self.init_ssthresh
        self.seq = 0

    # =========================
    # RESET (IMPORTANT)
    # =========================
    def reset(self):
        self.cwnd = self.init_cwnd
        self.ssthresh = self.init_ssthresh
        self.seq = 0

    def send(self, time):

        packets = []

        for _ in range(int(self.cwnd)):
            packets.append(Packet(self.seq, time))
            self.seq += 1

        return packets

    @abstractmethod
    def on_ack(self, ack):
        pass

    @abstractmethod
    def on_loss(self):
        pass
