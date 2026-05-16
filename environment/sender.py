from environment import Packet


class Sender:
    def __init__(self):

        self.cwnd = 10.0

        self.ssthresh = 32.0

        self.next_seq = 0

    def apply_action(self, action):

        # RL action

        if action == 0:
            self.cwnd *= 0.7

        elif action == 2:
            self.cwnd *= 1.2

        self.cwnd = max(1.0, self.cwnd)

    def send_packets(self):

        packets = []

        send_num = int(self.cwnd)

        for _ in range(send_num):
            pkt = Packet(self.next_seq)

            packets.append(pkt)

            self.next_seq += 1

        return packets
