from .sender import BaseSender


class AgentSender(BaseSender):
    def apply_action(self, action):
        """
        action:
            0 -> decrease
            1 -> hold
            2 -> increase
        """

        if action == 0:
            self.cwnd *= 0.7

        elif action == 2:
            self.cwnd *= 1.2

        self.cwnd = max(1.0, self.cwnd)

    def on_ack(self, ack_count):
        pass

    def on_loss(self):
        pass
