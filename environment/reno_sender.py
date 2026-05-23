from .sender import BaseSender

# ============================================================
# Reno Sender
# ============================================================


class RenoSender(BaseSender):
    def on_ack(self, ack):

        if self.cwnd < self.ssthresh:
            self.cwnd *= 2

        else:
            self.cwnd += 1 / self.cwnd

    def on_loss(self):

        self.ssthresh = max(2, self.cwnd / 2)

        self.cwnd = self.ssthresh
