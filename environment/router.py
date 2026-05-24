import numpy as np


class Router:
    def __init__(self, rng=None):

        # ===== local RNG (IMPORTANT) =====
        self.rng = rng or np.random.default_rng()

        self.base_bandwidth = 80
        self.queue_limit = 300
        self.queue = []

        self.wireless_loss = 0.02
        self.avg_packet_size = 40

        self.congestion_history = []

    def reset(self):
        self.queue = []
        self.congestion_history = []

    def transmit(self, packets, time):

        delivered = []
        congestion_drop = 0
        wireless_drop = 0

        # ================= enqueue =================
        for p in packets:
            if len(self.queue) < self.queue_limit:
                self.queue.append(p)
            else:
                congestion_drop += 1

        # ================= bandwidth fluctuation =================
        fading = self.rng.uniform(0.85, 1.15)

        udp_traffic = self.rng.integers(5, 15) * 8

        current_bandwidth = int(self.base_bandwidth * fading - udp_traffic)
        current_bandwidth = max(10, min(80, current_bandwidth))

        self.congestion_history.append(current_bandwidth)

        if len(self.congestion_history) > 5:
            self.congestion_history.pop(0)

        bandwidth = int(np.mean(self.congestion_history))

        # ================= transmit =================
        transmit_time = self.avg_packet_size / max(1, bandwidth)

        send_num = min(bandwidth, len(self.queue))

        for _ in range(send_num):
            pkt = self.queue.pop(0)

            # wireless loss
            if self.rng.random() < self.wireless_loss:
                wireless_drop += 1
                continue

            delivered.append(pkt)

        # ================= queue delay =================
        queue_delay = len(self.queue) * transmit_time

        total_sent = len(packets)
        loss_rate = (congestion_drop + wireless_drop) / max(1, total_sent)

        return {
            "delivered": delivered,
            "congestion_drop": congestion_drop,
            "wireless_drop": wireless_drop,
            "queue_delay": queue_delay,
            "transmit_time": transmit_time,
            "bandwidth": bandwidth,
            "loss_rate": loss_rate,
            "queue_length": len(self.queue),
        }
