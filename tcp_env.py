import random

import numpy as np


class WirelessTCPEnv:
    def __init__(self):

        self.max_steps = 30

        self.reset()

    def reset(self):

        # congestion window
        self.cwnd = 10.0

        # base RTT (ms)
        self.base_rtt = 50.0

        # current RTT
        self.rtt = self.base_rtt

        # packet loss probability
        self.loss_rate = 0.01

        # link bandwidth
        self.bandwidth = 100.0

        self.step_count = 0

        return self._get_state()

    def _get_state(self):

        return np.array(
            [self.cwnd, self.rtt, self.loss_rate, self.bandwidth], dtype=np.float32
        )

    def step(self, action):
        """
        action:
            0 -> decrease cwnd
            1 -> hold
            2 -> increase cwnd
        """

        # ===== congestion control action =====

        if action == 0:
            self.cwnd *= 0.7

        elif action == 2:
            self.cwnd *= 1.2

        # avoid too small
        self.cwnd = max(1.0, self.cwnd)

        # ===== wireless random loss =====

        wireless_noise = random.uniform(0.0, 0.05)

        effective_loss = self.loss_rate + wireless_noise

        # ===== RTT model =====
        # higher cwnd => more queue delay

        queue_delay = self.cwnd * 0.5

        self.rtt = self.base_rtt + queue_delay

        # ===== throughput model =====
        # simplified TCP throughput estimation

        throughput = self.cwnd * (1.0 - effective_loss)

        # ===== reward =====

        reward = throughput - 0.1 * self.rtt - 20 * effective_loss

        self.step_count += 1

        done = self.step_count >= self.max_steps

        info = {
            "throughput": throughput,
            "loss": effective_loss,
            "queue_delay": queue_delay,
        }

        return (self._get_state(), reward, done, info)
