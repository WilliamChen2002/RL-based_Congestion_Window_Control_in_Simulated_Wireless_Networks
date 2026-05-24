import random
import numpy as np


class Router:
    def __init__(self):
        self.base_bandwidth = 80  # 基礎頻寬（調整為更合理的值）
        self.queue_limit = 300  # 增加 queue size，讓擁塞較不易發生
        self.queue = []

        self.wireless_loss = 0.02  # 降低無線丟包率（原本 0.05 偏高）
        self.avg_packet_size = 40  # 平均封包大小（bytes）

        self.congestion_history = []  # 用來平滑頻寬變化

    def reset(self):
        """重置 Router 狀態"""
        self.queue = []
        self.congestion_history = []

    def transmit(self, packets, time):
        delivered = []
        congestion_drop = 0
        wireless_drop = 0

        # ==================== Enqueue ====================
        for p in packets:
            if len(self.queue) < self.queue_limit:
                self.queue.append(p)
            else:
                congestion_drop += 1

        # ==================== Bandwidth with smooth fading ====================
        # 平滑的頻寬變化（避免劇烈震盪）
        fading = random.uniform(0.85, 1.15)  # 範圍縮小，更穩定

        # 模擬背景流量（UDP）
        udp_traffic = random.randint(5, 15) * 8

        current_bandwidth = int(self.base_bandwidth * fading - udp_traffic)
        current_bandwidth = max(10, min(80, current_bandwidth))  # 限制範圍

        # 平滑處理（exponential moving average）
        self.congestion_history.append(current_bandwidth)
        if len(self.congestion_history) > 5:
            self.congestion_history.pop(0)

        bandwidth = int(np.mean(self.congestion_history))

        # ==================== Transmit ====================
        transmit_time = self.avg_packet_size / max(1, bandwidth)

        # 這次可傳輸的封包數
        send_num = min(bandwidth, len(self.queue))

        for _ in range(send_num):
            pkt = self.queue.pop(0)

            # Wireless loss
            if random.random() < self.wireless_loss:
                wireless_drop += 1
                continue

            delivered.append(pkt)

        # Queue delay
        queue_delay = len(self.queue) * transmit_time

        # ==================== 統計 ====================
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
