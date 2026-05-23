from .sender import BaseSender

# ============================================================
# Reno Sender
# ============================================================


class RenoSender(BaseSender):
    def on_ack(self, ack_count):

        for _ in range(ack_count):
            # Slow Start
            if self.cwnd < self.ssthresh:
                self.cwnd *= 2

            # Congestion Avoidance
            else:
                self.cwnd += 1 / self.cwnd

    def on_loss(self):

        self.ssthresh = max(2.0, self.cwnd / 2)

        self.cwnd = self.ssthresh
