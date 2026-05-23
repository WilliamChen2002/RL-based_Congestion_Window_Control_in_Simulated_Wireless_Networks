from abc import ABC, abstractmethod

from .packet import Packet

# ============================================================
# Base Sender
# ============================================================


class BaseSender(ABC):
    def __init__(self):

        self.cwnd = 10.0

        self.ssthresh = 32.0

        self.next_seq = 0

    def generate_packets(self, current_time):

        packets = []

        send_num = max(1, int(self.cwnd))

        for _ in range(send_num):
            pkt = Packet(seq_id=self.next_seq, created_time=current_time)

            packets.append(pkt)

            self.next_seq += 1

        return packets

    @abstractmethod
    def on_ack(self, ack_count):
        pass

    @abstractmethod
    def on_loss(self):
        pass
