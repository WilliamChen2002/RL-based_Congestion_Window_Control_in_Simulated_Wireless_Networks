from abc import ABC, abstractmethod

from .packet import Packet

# ============================================================
# Base Sender
# ============================================================


class BaseSender(ABC):
    def __init__(self):

        self.cwnd = 10.0

        self.ssthresh = 32.0

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
