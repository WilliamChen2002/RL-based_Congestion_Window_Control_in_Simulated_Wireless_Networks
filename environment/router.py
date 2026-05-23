import random


class Router:
    def __init__(self):

        self.base_bandwidth = 100

        self.queue_limit = 100

        self.queue = []

        self.wireless_loss = 0.05

        self.packet_size = random.randint(20, 60)  # TCP packet size

    def transmit(self, packets, time):

        delivered = []

        congestion_drop = 0

        wireless_drop = 0

        # enqueue
        for p in packets:
            if len(self.queue) < self.queue_limit:
                self.queue.append(p)

            else:
                congestion_drop += 1

        # ===== wireless fading (important upgrade) =====
        fading = random.uniform(0.5, 1.5)

        # UDP packet size(bytes)
        udp_size = 8
        udp_sender = random.randint(10, 20) * udp_size

        bandwidth = max(1, int(self.base_bandwidth * fading))

        bandwidth = max(1, int(self.base_bandwidth - udp_sender))

        # ===== transmit time model =====
        transmit_time = self.packet_size / bandwidth

        send_num = min(bandwidth, len(self.queue))

        for _ in range(send_num):
            pkt = self.queue.pop(0)

            # wireless loss
            if random.random() < self.wireless_loss:
                wireless_drop += 1

                continue

            delivered.append(pkt)

        queue_delay = len(self.queue) * transmit_time

        return {
            "delivered": delivered,
            "congestion_drop": congestion_drop,
            "wireless_drop": wireless_drop,
            "queue_delay": queue_delay,
            "transmit_time": transmit_time,
            "bandwidth": bandwidth,
        }
