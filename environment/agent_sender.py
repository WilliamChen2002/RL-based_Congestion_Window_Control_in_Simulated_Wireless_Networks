from .sender import BaseSender


class AgentSender(BaseSender):
    def apply(self, action=1, cwnd=None, coef=None):

        if cwnd is not None and coef is not None:
            self.cwnd = cwnd * coef

        elif coef is not None:
            self.cwnd *= coef

        elif cwnd is not None:
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

    # =========================
    # RESET FIX
    # =========================
    def reset(self):
        super().reset()
