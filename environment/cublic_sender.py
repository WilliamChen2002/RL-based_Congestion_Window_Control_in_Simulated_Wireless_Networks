from .sender import BaseSender


class CubicSender(BaseSender):
    def __init__(self):

        super().__init__()

        self.Wmax = self.cwnd

        self.t = 0

        self.C = 0.4

        self.beta = 0.7

    def on_ack(self, ack):

        self.t += 1

        self.cwnd = self.C * (self.t**3) + self.Wmax

        self.cwnd = max(1.0, self.cwnd)

    def on_loss(self):

        self.Wmax = self.cwnd

        self.cwnd *= self.beta

        self.t = 0
