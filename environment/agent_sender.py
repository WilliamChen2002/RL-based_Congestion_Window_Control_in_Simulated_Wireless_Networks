from .sender import BaseSender


class AgentSender(BaseSender):
    def apply(self, action=1, cwnd=None):
        if cwnd is not None:
            self.cwnd = cwnd
        else:
            if action == 0:
                self.cwnd *= 0.7

            elif action == 2:
                self.cwnd *= 1.2

        self.cwnd = max(1.0, self.cwnd)

    def on_ack(self, ack):
        pass

    def on_loss(self):
        pass
