from .sender import BaseSender


class CubicSender(BaseSender):
    def __init__(self):

        super().__init__()

        self.C = 0.4

        self.beta = 0.7

        self.Wmax = self.cwnd

        self.epoch_time = 0

    def on_ack(self, ack_count):

        self.epoch_time += 1

        K = ((self.Wmax * (1 - self.beta)) / self.C) ** (1 / 3)

        self.cwnd = self.C * ((self.epoch_time - K) ** 3) + self.Wmax

        self.cwnd = max(1.0, self.cwnd)

    def on_loss(self):

        self.Wmax = self.cwnd

        self.cwnd *= self.beta

        self.cwnd = max(1.0, self.cwnd)

        self.epoch_time = 0
