class Receiver:
    def __init__(self):

        self.last_update_time = 0

    def receive(self, packets, current_time):

        ack = len(packets)

        if ack > 0:
            self.last_update_time = packets[-1].created_time

        aoi = current_time - self.last_update_time

        return ack, aoi
