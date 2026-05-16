import random


class Router:
    def __init__(self):

        self.bandwidth = 20

        self.queue_size = 50

        self.base_delay = 50

        self.wireless_loss = 0.05

        self.queue = []

    def transmit(self, packets):

        delivered = []

        dropped = 0

        # enqueue
        for pkt in packets:
            if len(self.queue) < self.queue_size:
                self.queue.append(pkt)

            else:
                dropped += 1

        # bandwidth limit
        transmit_num = min(self.bandwidth, len(self.queue))

        for _ in range(transmit_num):
            pkt = self.queue.pop(0)

            # wireless random drop
            if random.random() < self.wireless_loss:
                dropped += 1

                continue

            delivered.append(pkt)

        queue_delay = len(self.queue) * 0.5

        rtt = self.base_delay + queue_delay

        return delivered, dropped, rtt
