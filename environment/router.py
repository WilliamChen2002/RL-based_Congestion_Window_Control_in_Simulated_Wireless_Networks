import random


class Router:
    def __init__(
        self, bandwidth=20, queue_limit=100, wireless_loss=0.05, base_delay=50
    ):

        self.bandwidth = bandwidth

        self.queue_limit = queue_limit

        self.wireless_loss = wireless_loss

        self.base_delay = base_delay

        self.queue = []

    def transmit(self, packets):

        delivered = []

        congestion_drop = 0

        wireless_drop = 0

        # enqueue
        for pkt in packets:
            if len(self.queue) < self.queue_limit:
                self.queue.append(pkt)

            else:
                congestion_drop += 1

        # transmit
        send_num = min(self.bandwidth, len(self.queue))

        for _ in range(send_num):
            pkt = self.queue.pop(0)

            # wireless random loss
            if random.random() < self.wireless_loss:
                wireless_drop += 1

                continue

            delivered.append(pkt)

        queue_delay = len(self.queue) * 0.5

        rtt = self.base_delay + queue_delay

        return {
            "delivered": delivered,
            "congestion_drop": congestion_drop,
            "wireless_drop": wireless_drop,
            "queue_delay": queue_delay,
            "rtt": rtt,
        }
